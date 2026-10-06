// Writes a random-weight GPT-2 checkpoint, token shards and a tokenizer in
// llm.c's file formats, for runs that need a model of a given size but not its
// trained weights (memory and speed measurements).
//
// usage: make_random OUT_DIR maxT V Vp L NH C NTOKENS
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 1337;

static uint32_t next_u32(void) {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return (uint32_t)((rng * 0x2545F4914F6CDD1Dull) >> 32);
}

static FILE* open_out(const char* dir, const char* name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE* f = fopen(path, "wb");
    if (f == NULL) { perror(path); exit(1); }
    return f;
}

int main(int argc, char** argv) {
    if (argc != 9) { fprintf(stderr, "usage: make_random OUT_DIR maxT V Vp L NH C NTOKENS\n"); return 1; }
    int maxT = atoi(argv[2]), V = atoi(argv[3]), Vp = atoi(argv[4]), L = atoi(argv[5]);
    int NH = atoi(argv[6]), C = atoi(argv[7]);
    long ntok = atol(argv[8]);
    long n = (long)Vp * C + (long)maxT * C + (long)L * (12L * C * C + 13L * C) + 2L * C;

    int header[256] = {0};
    header[0] = 20240326; header[1] = 3; header[2] = maxT; header[3] = V;
    header[4] = L; header[5] = NH; header[6] = C; header[7] = Vp;
    FILE* f = open_out(argv[1], "model.bin");
    fwrite(header, sizeof(int), 256, f);
    float buf[4096];
    for (long i = 0; i < n; i += 4096) {
        long m = n - i < 4096 ? n - i : 4096;
        for (long j = 0; j < m; j++) { buf[j] = ((next_u32() >> 8) / 16777216.0f - 0.5f) * 0.04f; }
        fwrite(buf, sizeof(float), m, f);
    }
    fclose(f);

    const char* shards[2] = { "train_tokens.bin", "val_tokens.bin" };
    for (int s = 0; s < 2; s++) {
        int data_header[256] = {0};
        data_header[0] = 20240520; data_header[1] = 1; data_header[2] = (int)ntok;
        f = open_out(argv[1], shards[s]);
        fwrite(data_header, sizeof(int), 256, f);
        for (long i = 0; i < ntok; i++) {
            uint16_t token = (uint16_t)(next_u32() % V);
            fwrite(&token, sizeof(uint16_t), 1, f);
        }
        fclose(f);
    }

    uint32_t tok_header[256] = {0};
    tok_header[0] = 20240328; tok_header[1] = 2; tok_header[2] = V; tok_header[3] = V - 1;
    f = open_out(argv[1], "tokenizer.bin");
    fwrite(tok_header, sizeof(uint32_t), 256, f);
    for (int i = 0; i < V; i++) {
        char text[16];
        snprintf(text, sizeof text, " w%d", i);
        unsigned char length = (unsigned char)strlen(text);
        fwrite(&length, 1, 1, f);
        fwrite(text, 1, length, f);
    }
    fclose(f);
    printf("num_parameters: %ld\n", n);
    return 0;
}
