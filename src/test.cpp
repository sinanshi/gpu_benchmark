#include <benchmark/benchmark.h>
//#include <stdlib.h> 

#define float float


static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
// Register the function as a benchmark
//BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state)
std::string copy(x);
}
//BENCHMARK(BM_StringCopy);

/*

static void BM_memcpy(benchmark::State& state) {
char* src = new char[state.range(0)];
char* dst = new char[state.range(0)];
memset(src, 'x', state.range(0));
for (auto _ : state)
memcpy(dst, src, state.range(0));
state.SetBytesProcessed(int64_t(state.iterations()) *
int64_t(state.range(0)));
delete[] src;
delete[] dst;
}
BENCHMARK(BM_memcpy)->Arg(8)->Arg(64)->Arg(512)->Arg(1<<10)->Arg(8<<10);


static void BM_StringCompare(benchmark::State& state) {
std::string s1(state.range(0), '-');
std::string s2(state.range(0), '-');
for (auto _ : state) {
benchmark::DoNotOptimize(s1.compare(s2));
}
state.SetComplexityN(state.range(0));
}
BENCHMARK(BM_StringCompare)
->RangeMultiplier(2)->Range(1<<10, 1<<18)->Complexity(benchmark::oN);
*/


static void BM_simonie(benchmark::State& state) {
  int N = state.range(0); 
  float c1 = 10;
  float c2 = 10;
  float *a = new float[N];
  float *b = new float[N];
  float *result = new float[N];
 
  for (auto _ : state) {
    for (auto i = 0; i < N; ++i) {
      a[i] = 0.1; 
      b[i] = 0.2;
      result[i] = 0.5;
    }


    for (int i = 0; i < N; ++i) {
      result[i] = c1 + a[i] * c2;

    }
  }
  delete[] a;
  delete[] b;
  delete[] result;


}
BENCHMARK(BM_simonie) -> Arg(1e5) -> Arg(1e6);



BENCHMARK_MAIN();
