#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_set>

#include <omp.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/rair/IndexReIVF.h>
#include <faiss/utils/distances.h>

using namespace faiss;
using namespace faiss::rair;

IndexReIVF::IndexReIVF(Index* quantizer, size_t d, size_t nlist,
                       size_t code_size, size_t nrep, MetricType metric):
                       IndexIVF(quantizer, d, nlist, code_size, metric),
                       nrep(nrep) {
  FAISS_THROW_IF_NOT_MSG(nrep==2, "IndexReIVF supports nrep==2 only.");
}

void IndexReIVF::cluster_assign(idx_t n, const float* x, idx_t* labels,
                                idx_t k) const {
  FAISS_THROW_IF_NOT_MSG(k==(idx_t)nrep, "IndexReIVF supports k==nrep only.");
  FAISS_THROW_IF_NOT(labels);

  if (redund_strategy == 0) {
    quantizer->assign(n, x, labels, k);

  } else if (redund_strategy == 1 || redund_strategy == 2) {
    idx_t k_ = 5*k; ///< number of centroid candidates
    std::unique_ptr<float[]> distances_(new float[n*k_]); ///< L2 sqr of residuals
    std::unique_ptr<idx_t[]> labels_(new idx_t[n*k_]);
    std::unique_ptr<float[]> recons_(new float[n*k_*d]);
    quantizer->search_and_reconstruct(n, x, k_, distances_.get(), labels_.get(),
                                      recons_.get());

    /// TODO: optimization
    std::unique_ptr<float[]> loss(new float[n*k_]);
    size_t sec_choice_cnt[10] = {0};
    float sum_nn_dis = 0.0f;
    float sum_loss = 0.0f;
    for (int i = 0; i < n; i ++) {
      labels[i*k] = labels_[i*k_]; ///< first choice
      float loss_min = distances_[i*k_] * 10000; ///< a big enough init value

      idx_t sec_choice_id = 0;
      /// compute residual
      for (int j = 0; j < k_; j ++) {
        for (int l = 0; l < d; l ++) {
          recons_[(i*k_+j)*d+l] -= x[i*d+l];
        }
      }
      /// compute loss
      for (int j = strict_redund_mode; j < k_; j ++) {
        float inner_prod = fvec_inner_product(recons_.get()+((i*k_+j)*d),
                                              recons_.get()+((i*k_+0)*d), d);
        if (redund_strategy == 1) { ///< RAIR
          loss[i*k_+j] = distances_[i*k_+j] + lambda * inner_prod;
        } else { ///< redund_strategy == 2, SOAR
          loss[i*k_+j] = distances_[i*k_+j] + lambda * inner_prod * inner_prod
              / distances_[i*k_];
        }
        /// Trick: only work for k=2
        if (loss[i*k_+j] < loss_min) {
          loss_min = loss[i*k_+j];
          labels[i*k+1] = labels_[i*k_+j]; ///< second choice
          if (j == 0) labels[i*k+1] = -1;
          sec_choice_id = j;
        }
      }
      sec_choice_cnt[sec_choice_id] ++;
      sum_nn_dis += distances_[i*k_];
      sum_loss += loss_min;
    }

  } else {
    FAISS_THROW_IF_NOT_MSG(false, "Bad redundant strategy");
  }
}

void IndexReIVF::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
  std::unique_ptr<idx_t[]> coarse_idx(new idx_t[nrep*n]);
  cluster_assign(n, x, coarse_idx.get(), nrep);
  add_core(n, x, xids, coarse_idx.get());
}

void IndexReIVF::add_core(idx_t n, const float* x, const idx_t* xids,
                          const idx_t* coarse_idx,
                          void* inverted_list_context) {
  /// do some blocking to avoid excessive allocs
  idx_t bs = 65536;
  if (n > bs) {
    for (idx_t i0 = 0; i0 < n; i0 += bs) {
      idx_t i1 = std::min(n, i0 + bs);
      add_core(i1-i0, x+i0*d, xids ? xids+i0 : nullptr, coarse_idx+i0*nrep,
               inverted_list_context);
    }
    return;
  }
  FAISS_THROW_IF_NOT(coarse_idx);
  FAISS_THROW_IF_NOT(is_trained);
  direct_map.check_can_add(xids);

  std::unique_ptr<uint8_t[]> codes(new uint8_t[n * code_size]);
  encode_vectors(n, x, coarse_idx, codes.get());

  /// no use by default
  DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int rank = omp_get_thread_num();

    /// each thread takes care of a subset of lists
    for (idx_t i = 0; i < n; i ++) {
      for (size_t j = 0; j < nrep; j ++) {
        idx_t list_no = coarse_idx[nrep*i+j];

        if (list_no >= 0 && list_no % nt == rank) {
          idx_t id = xids ? xids[i] : ntotal + i;
          size_t ofs = invlists->add_entry(
              list_no, id, codes.get() + i * code_size, inverted_list_context
              );
          dm_adder.add(i, list_no, ofs);
        } else if (rank == 0 && list_no == -1) {
          dm_adder.add(i, -1, 0);
        }
      }
    }
  }

  ntotal += n;
}

void
IndexReIVF::search_preassigned(idx_t n, const float* x, idx_t k,
                               const idx_t* assign,
                               const float* centroid_dis, float* distances,
                               idx_t* labels, bool store_pairs,
                               const IVFSearchParameters* params,
                               IndexIVFStats* stats) const {
  if (omit_redund_visit) {
    IndexIVF::search_preassigned(n, x, k, assign, centroid_dis, distances,
                                 labels, store_pairs, params, stats);
    return;
  }

  idx_t k_ = nrep * k;
  std::unique_ptr<float[]> distances_(new float[n*k_]);
  std::unique_ptr<idx_t[]> labels_(new idx_t[n*k_]);

  IndexIVF::search_preassigned(n, x, k_, assign, centroid_dis,
                               distances_.get(), labels_.get(),
                               store_pairs, params, stats);

  /// deduplicate the same label
  for (idx_t i = 0; i < n; i ++) {
    idx_t cur = 0, cnt = 0;
    while (cnt < k) {
      FAISS_ASSERT(cur < k_);
      if (cnt == 0 || labels_[i*k_+cur] != labels[i*k+cnt-1]) {
        distances[i*k+cnt] = distances_[i*k_+cur];
        labels[i*k+cnt] = labels_[i*k_+cur];
        cnt ++;
      }
      cur ++;
    }
  }
}

void
IndexReIVF::update_vectors(int nv, const idx_t* idx, const float* v) {
  FAISS_THROW_MSG("not implemented");
}

std::atomic<int> faiss::rair::omit_ndis = 0;
