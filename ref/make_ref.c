// Writes a random-weight GPT-2 checkpoint and a debug state in llm.c's file
// formats, using llm.c's own forward, backward and AdamW as the reference.
// The Makefile builds it without FMA contraction, so the C arithmetic matches
// Vx operation for operation.
#define TESTING
#include "train_gpt2.c"

static uint64_t rng = 1337;

static uint32_t next_u32(void) {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return (uint32_t)((rng * 0x2545F4914F6CDD1Dull) >> 32);
}

static float next_f32(void) { return (next_u32() >> 8) / 16777216.0f; }

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: make_ref <out-dir>\n"); return 1; }
    char path[1024];
    // A small GPT-2: V is not a multiple of 128 so the padded vocab is used.
    int maxT = 64, V = 500, Vp = 512, L = 2, NH = 4, C = 64, B = 4, T = 16;

    int header[256] = {0};
    header[0] = 20240326; header[1] = 3; header[2] = maxT; header[3] = V;
    header[4] = L; header[5] = NH; header[6] = C; header[7] = Vp;
    GPT2Config config = { maxT, V, Vp, L, NH, C };
    size_t sizes[NUM_PARAMETER_TENSORS];
    fill_in_parameter_sizes(sizes, config);
    size_t n = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) { n += sizes[i]; }
    float* params = (float*)mallocCheck(n * sizeof(float));
    for (size_t i = 0; i < n; i++) { params[i] = (next_f32() - 0.5f) * 0.2f; }
    // Zero the padded vocab rows, as llm.c's export does.
    for (int v = V; v < Vp; v++) { for (int c = 0; c < C; c++) { params[v * C + c] = 0.0f; } }
    snprintf(path, sizeof path, "%s/ref_model.bin", argv[1]);
    FILE* f = fopenCheck(path, "wb");
    fwrite(header, sizeof(int), 256, f);
    fwrite(params, sizeof(float), n, f);
    fcloseCheck(f);

    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    for (int i = 0; i < B * T; i++) { x[i] = next_u32() % V; y[i] = next_u32() % V; }

    GPT2 model;
    gpt2_build_from_checkpoint(&model, path);
    gpt2_forward(&model, x, y, B, T);
    gpt2_zero_grad(&model);
    gpt2_backward(&model);

    int state[256] = {0};
    state[0] = 20240327; state[1] = 2; state[2] = B; state[3] = T;
    snprintf(path, sizeof path, "%s/ref_state.bin", argv[1]);
    f = fopenCheck(path, "wb");
    fwrite(state, sizeof(int), 256, f);
    fwrite(x, sizeof(int), B * T, f);
    fwrite(y, sizeof(int), B * T, f);
    // The PyTorch state stores unpadded logits, so drop the padding here too.
    for (int bt = 0; bt < B * T; bt++) { fwrite(model.acts.logits + bt * Vp, sizeof(float), V, f); }
    fwrite(&model.mean_loss, sizeof(float), 1, f);
    fwrite(model.grads_memory, sizeof(float), model.num_parameters, f);
    fcloseCheck(f);

    // The ten training losses test_gpt2 compares against.
    float losses[10];
    for (int step = 0; step < 10; step++) {
        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
        losses[step] = model.mean_loss;
        printf("step %d: loss %.9g\n", step, model.mean_loss);
    }
    snprintf(path, sizeof path, "%s/ref_losses.bin", argv[1]);
    f = fopenCheck(path, "wb");
    fwrite(losses, sizeof(float), 10, f);
    fcloseCheck(f);
    return 0;
}
