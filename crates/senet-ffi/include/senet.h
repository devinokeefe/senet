/* senet.h: the C ABI of the Senet engine library, senet_ffi (crates/senet-ffi).
 *
 * python/senet/_lib.py is its Python binding. python/tests/test_abi.py checks that this
 * header, the library's Rust source and the Python declarations agree: the functions,
 * their parameter and result types, the constants and the layout of senet_move.
 *
 * Positions are two square masks (bit i = square i, squares 1 to 30) from the point of
 * view of the player about to throw.
 *
 * Results: integer results are >= 0 on success. On failure they are SENET_INVALID (a bad
 * argument: an invalid position or an empty board, an invalid throw, a finished game where
 * moves are asked for, an unknown bot spec, no games to play, a NULL or misaligned pointer,
 * a string that is not UTF-8, or a closed handle), SENET_UNAVAILABLE (a database or network
 * the call needs is not loaded or does not cover the position; perfect play and
 * senet_quality need the complete database) or, for senet_bot_choose, SENET_NO_MOVE. Value
 * results (double) are probabilities in [0, 1], or -1.0 / -2.0 with the meanings of
 * SENET_INVALID / SENET_UNAVAILABLE. A call that fails writes none of its outputs.
 *
 * Messages: every failure stores a message in a text slot of the calling thread, and
 * senet_match, senet_quality and senet_build_info store their JSON result there.
 * senet_last_text copies it out; it stays until the thread's next call that stores text.
 *
 * Pointers: each pointer must be aligned for its type, and an output must not overlap any
 * other argument. The library refuses NULL and misaligned pointers, and arrays of more than
 * PTRDIFF_MAX bytes (SENET_INVALID), but cannot check sizes: an array must hold the number
 * of elements the call is given. A buffer of zero elements may be NULL. Strings are
 * NUL-terminated UTF-8 and are read during the call only.
 *
 * Handles: senet_ctx_new returns a handle owning a database, a network, both or neither.
 * Any number of threads may use a handle at once, and close it while others use it: its
 * resources are released when the last call using them returns. Only senet_ctx_free must
 * wait until no other call can use the handle.
 *
 * A panic inside the library is caught and reported as a SENET_INVALID failure.
 */
#ifndef SENET_H
#define SENET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The ABI this header describes: senet_abi_version() of a library that implements it. */
#define SENET_ABI_VERSION 6

#define SENET_INVALID (-1)
#define SENET_UNAVAILABLE (-2)
#define SENET_NO_MOVE (-3)

/* senet_move.kind */
#define SENET_MOVE_STEP 0  /* to an empty square */
#define SENET_MOVE_SWAP 1  /* swapping places with an unprotected opposing piece */
#define SENET_MOVE_OFF 2   /* bearing the piece off */
#define SENET_MOVE_WATER 3 /* onto the House of Water, which sends the piece back */

/* A legal move (12 bytes, aligned to 4). */
typedef struct senet_move {
    uint8_t from;      /* the square the piece leaves */
    uint8_t to;        /* where it comes to rest: 31 if borne off; for WATER, where it is sent back to */
    uint8_t kind;      /* SENET_MOVE_STEP, SENET_MOVE_SWAP, SENET_MOVE_OFF or SENET_MOVE_WATER */
    uint8_t back;      /* 1 for a backward move, else 0 */
    uint32_t me_after; /* the position after the move, still from the mover's view */
    uint32_t opp_after;
} senet_move;

/* An engine handle (opaque). */
typedef struct senet_ctx senet_ctx;

/* The ABI version of the library. */
uint32_t senet_abi_version(void);

/* Copies the calling thread's text slot into `out` as a NUL-terminated string, cut to at
 * most `cap` - 1 bytes but never inside a UTF-8 character; `out` may be NULL to ask the
 * size. Returns the size needed for all of it (its length + 1). */
size_t senet_last_text(char *out, size_t cap);

/* How the library was built, as JSON in the text slot: `version`, `target`, `optimized`
 * and the `cpu_features` it requires. Returns the JSON's length. */
int64_t senet_build_info(void);

/* The most legal moves a throw can have: a capacity that always fits. */
uint32_t senet_max_moves(void);

/* The length of the network's feature vector. */
uint32_t senet_n_features(void);

/* Writes White's view of the opening position. Returns 0. */
int32_t senet_start(uint32_t *me, uint32_t *opp);

/* The legal moves for throw `t` (1 to 5) in a game in progress: writes the first `cap` of
 * them to `out` and returns how many there are. */
int32_t senet_gen_moves(uint32_t me, uint32_t opp, uint8_t t, senet_move *out, size_t cap);

/* Writes the database layer (`w`, `b`) and index `idx` of a game in progress. Returns 0. */
int32_t senet_index_of(uint32_t me, uint32_t opp, uint32_t *w, uint32_t *b, uint64_t *idx);

/* Writes the position with index `idx` in layer (`w`, `b`). Returns 0. */
int32_t senet_position_of(uint32_t w, uint32_t b, uint64_t idx, uint32_t *me, uint32_t *opp);

/* The number of positions in layer (`w`, `b`), or 0 if there is no such layer. */
uint64_t senet_layer_size(uint32_t w, uint32_t b);

/* The hand-made heuristic's estimate of P(player to throw wins). */
double senet_heuristic(uint32_t me, uint32_t opp);

/* Writes the network's senet_n_features() input features for a position. Returns 0. */
int32_t senet_features(uint32_t me, uint32_t opp, float *out);

/* A handle owning the database in directory `db_dir` and the network in file `net_path`,
 * either of which may be NULL; NULL on failure. */
senet_ctx *senet_ctx_new(const char *db_dir, const char *net_path);

/* Closes a handle: later calls with it fail. Closing it again does nothing. */
void senet_ctx_close(const senet_ctx *ctx);

/* Frees a handle, closing it first. */
void senet_ctx_free(senet_ctx *ctx);

/* The perfect-play value of a position, from the database. */
double senet_db_value(const senet_ctx *ctx, uint32_t me, uint32_t opp);

/* The perfect-play values of `n` positions, computed in parallel: each -1 if the position is
 * invalid, -2 if the database does not cover it. Returns 0. */
int32_t senet_db_values(const senet_ctx *ctx, const uint32_t *me, const uint32_t *opp, size_t n, float *out);

/* The network's estimate of P(player to throw wins). */
double senet_net_value(const senet_ctx *ctx, uint32_t me, uint32_t opp);

/* The mover's win probability after each legal move for throw `t`, as bot `spec` judges it:
 * writes the first `cap` values to `out` and returns how many legal moves there are. */
int32_t senet_move_values(const senet_ctx *ctx, const char *spec, uint32_t me, uint32_t opp, uint8_t t, double *out, size_t cap);

/* The index (into the legal moves) of the move bot `spec` plays for throw `t`, or
 * SENET_NO_MOVE if there is no legal move. */
int32_t senet_bot_choose(const senet_ctx *ctx, const char *spec, uint32_t me, uint32_t opp, uint8_t t, uint64_t seed);

/* Plays `pairs` pairs of games (`pairs` >= 1) between bots `a` and `b`, each pair with the
 * same throws and colours swapped: the result as JSON in the text slot. Returns its
 * length. */
int64_t senet_match(const senet_ctx *ctx, const char *a, const char *b, uint64_t pairs, uint64_t seed);

/* Plays `games` games (`games` >= 1) of bot `spec` against bot `vs` and measures the win
 * probability `spec` gives away per decision compared with perfect play: the result as JSON
 * in the text slot. Returns its length. */
int64_t senet_quality(const senet_ctx *ctx, const char *spec, const char *vs, uint64_t games, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* SENET_H */
