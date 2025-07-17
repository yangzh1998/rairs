#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <sys/stat.h>
#include <sys/time.h>
#include <unordered_set>
#include <faiss/IndexFlat.h>
#include <faiss/rair/IndexReIVFPQFastScan.h>
#include <faiss/IndexRefine.h>
#include <faiss/utils/distances.h>
#include <faiss/rair/impl/dedup_simd_result_handlers.h>
#include <omp.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
  FILE* f = fopen(fname, "r");
  if (!f) {
    fprintf(stderr, "could not open %s\n", fname);
    perror("");
    abort();
  }
  int d;
  fread(&d, 1, sizeof(int), f);
  assert((d > 0 && d < 1000000) || !"unreasonable dimension");
  fseek(f, 0, SEEK_SET);
  struct stat st;
  fstat(fileno(f), &st);
  size_t sz = st.st_size;
  assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
  size_t n = sz / ((d + 1) * 4);

  *d_out = d;
  *n_out = n;
  float* x = new float[n * (d + 1)];
  size_t nr = fread(x, sizeof(float), n * (d + 1), f);
  assert(nr == n * (d + 1) || !"could not read whole file");

  // shift array to remove row headers
  for (size_t i = 0; i < n; i ++)
    memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

  fclose(f);
  return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
  return (int*)fvecs_read(fname, d_out, n_out);
}

/**
 * recall a at b
 * pred & gt: size n*k
 */
float calc_recall(size_t n, size_t k, size_t a, size_t b,
                  faiss::idx_t* pred, faiss::idx_t* gt) {
  assert(a <= b && a <= 100 && b <= 1000);
  int cnt = 0;
  for (int i = 0; i < n; i ++) { // for each query
    // check if duplicated
    std::unordered_set<faiss::idx_t> pred_q;
    for (int j = 0; j < k; j ++) {
      if (pred[i * k + j] < 0) continue;
      if (pred_q.find(pred[i * k + j]) != pred_q.end()) {
        printf("ERROR: duplicate id in result.\n");
        assert(false);
      }
      pred_q.insert(pred[i * k + j]);
    }

    int cnt_ = 0;
    for (int j = 0; j < b; j ++) {
      for (int l = 0; l < a; l ++) {
        if (pred[i * k + j] == gt[i * k + l]) {
          cnt_ ++;
          break;
        }
      }
      if (cnt_ == a) break;
    }
    cnt += cnt_;
  }
  return (float)cnt / (float)n / a;
}

double elapsed() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

/// some test parameter
namespace {
bool use_seil = true;
} // anonymous namespace

int test_rair(const char* learn_file, const char* base_file,
              const char* query_file, const char* gnd_truth_file,
              size_t dim) {

  double t0 = elapsed();
  faiss::Index* index = nullptr;

  // Using IVF
  faiss::IndexIVF* indexIVF = nullptr;
  size_t nlist = 1024;

  faiss::Index* coarse_quantizer = new faiss::IndexFlat(dim);
  indexIVF = new faiss::rair::IndexReIVFPQFastScan(
      coarse_quantizer, dim, nlist, dim/2, 4
      );
  ((faiss::rair::IndexReIVFFastScan*)indexIVF)->block_pruning = use_seil;
  ((faiss::rair::IndexReIVF*)indexIVF)->redund_strategy = 1; // AIR
  ((faiss::rair::IndexReIVF*)indexIVF)->strict_redund_mode = 0; // Non-strict
  ((faiss::rair::IndexReIVF*)indexIVF)->lambda = 0.5;
  indexIVF->quantizer_trains_alone = 2;

  auto* indexRefine = new faiss::IndexRefineFlat(indexIVF);
  // Instead of 10-NN search with k_factor = 10, here we do 100-NN search with k_factor = 1
  indexRefine->k_factor = 1;
  index = indexRefine;

  size_t d, nt, nb, nq;
  
  // train
  printf("[%.3f s] Loading train set\n", elapsed() - t0);
  float* xt = fvecs_read(learn_file, &d, &nt); assert(d==dim);
  printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
  index->train(nt, xt);
  delete[] xt;

  // add
  printf("[%.3f s] Loading database\n", elapsed() - t0);
  float* xb = fvecs_read(base_file, &d, &nb); assert(d == dim);
  printf("[%.3f s] Indexing database, size %ld*%ld\n", elapsed() - t0, nb, d);
  index->add(nb, xb);
  delete[] xb;

  // load query
  printf("[%.3f s] Loading queries\n", elapsed() - t0);
  float* xq = fvecs_read(query_file, &d, &nq); assert(d == dim);
  size_t k;                // nb of results per query in the GT
  faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors
  printf("[%.3f s] Loading ground truth for %ld queries\n", elapsed() - t0, nq);
  
  // load ground-truth and convert int to long
  size_t nq2;
  int* gt_int = ivecs_read(gnd_truth_file, &k, &nq2);
  assert(nq2 == nq || !"incorrect nb of ground truth entries");
  gt = new faiss::idx_t[k * nq];
  for (int i = 0; i < nq * k; i ++) {
    gt[i] = gt_int[i];
  }
  delete[] gt_int;

  // output buffers
  faiss::idx_t* I = new faiss::idx_t[nq * k];
  float* D = new float[nq * k];
  
  // query
  omp_set_num_threads(4);
  printf("Threads: %d\n", omp_get_max_threads());
  faiss::indexIVF_stats.reset();
  faiss::rair::omit_ndis = 0;
  faiss::rair::dedupReservoirStat.reset();
  indexIVF->nprobe = 32;
  
  double search_start = elapsed();
  index->search(nq, xq, k, D, I);
  double search_end = elapsed();
  
  printf("[%.3f s] ndis: %zu, search_time: %.5lf s\n", elapsed() - t0,
         (faiss::indexIVF_stats.ndis - faiss::rair::omit_ndis),
         (search_end - search_start));
  printf("[%.3f s] Compute recalls\n", elapsed() - t0);
  printf("10@10 = %.4f\n", calc_recall(nq, k, 10, 10, I, gt));
  printf("[%.3f s] Closing\n", elapsed() - t0);

  delete[] I, D, xq, gt;
  delete index;

  return 0;
}

#define SIFT_DIR_PATH "<your path>"

const char* SIFT_LEARN_FILE = SIFT_DIR_PATH "sift_learn.fvecs";
const char* SIFT_BASE_FILE = SIFT_DIR_PATH "sift_base.fvecs";
const char* SIFT_QUERY_FILE = SIFT_DIR_PATH "sift_query.fvecs";
const char* SIFT_GND_TRUTH_FILE = SIFT_DIR_PATH "sift_groundtruth.ivecs";

int main() {
  return test_rair(SIFT_LEARN_FILE, SIFT_BASE_FILE,
                   SIFT_QUERY_FILE, SIFT_GND_TRUTH_FILE, 128);
}
