#!/usr/bin/env bash
#
# Rebuilds the Lifelong 3D Maze (`wmaze`) dataset from its two YouTube source videos,
# reproducing the exact recipe used for the paper:
#
#   1. download the two 10-hour "Windows 3D Maze" screensaver videos with yt-dlp
#   2. crop / rescale each to 64x64 @ 30fps, concatenate to one ~20h video
#   3. resample to 20fps and split into an 18h train mp4 and a ~2h test mp4
#   4. convert both mp4s to 500-frame .npy chunks with mp4_to_npy.py
#   5. lay the chunks out as <save_dir>/{train,test}/0/<chunk>.npy and set T_total=1000000
#      in config.json so the loader only ever touches the first 1M train frames
#
# Resulting stream layout (20 fps):
#   frames         0 - 1,000,000   train  (T_total; what the models are trained on)
#   frames 1,000,000 - 1,296,000   present in train/0 but never read (buffer between splits)
#   frames 1,296,000 - 1,439,400   test/0 (evaluations use the first 100k of these)
#
# The ffmpeg / yt-dlp invocations below follow the recipe in mp4_to_npy.py's docstring, with two
# reconstructions that were checked against the stored dataset:
#   * the two 10h clips are joined with a stream-copy concat, Video 1 first (the segment cut
#     lands at stream frame 719,475 = 35973.73 s x 20 fps, and the stream opens with Video 1);
#   * the docstring's "-aspect 300:240" step for Video 2 is "-s 300x240" here. Video 2 is a 16:9
#     4K upload, and -aspect only rewrites metadata, so the documented centre crop
#     (crop=240:240:30:0) can only produce the stored frames -- centred vanishing point, no
#     pillarbox edges, sharper than Video 1 -- if the pixels were first resized to 300x240.
#     Video 2's download resolution is inferred (high-res), not recorded; 720p is kept.
#
# Usage (from anywhere):
#   bash datasets/preprocess_wmaze.sh [SAVE_DIR]
#     SAVE_DIR  dataset directory the code reads from; defaults to datasets/windows_maze.
#               The downloads and intermediate mp4s are written into it while building and deleted
#               once the chunks are in place, so a finished build holds only {train,test}/0/*.npy
#               and config.json. Set KEEP_VIDEOS=1 to keep them (e.g. to inspect a rebuild).
#
# Requirements: a *current* yt-dlp (YouTube rejects releases that are more than a few months
# old; e.g. 2024.04.09 only returns storyboards -- run `yt-dlp -U`), ffmpeg + ffprobe with a VP9
# decoder, and a python with opencv-python + numpy (override the interpreter with
# PYTHON=/path/to/python). yt-dlp also needs a JavaScript runtime for YouTube: install deno
# (https://deno.land) somewhere in PATH and the script passes it to yt-dlp; without one YouTube
# throttles the downloads to ~50 KB/s (node >= 20 works too: YTDLP_ARGS="--js-runtimes node").
# From university / datacenter networks YouTube often answers "Sign in to confirm you're not a
# bot"; pass browser cookies through YTDLP_ARGS, e.g. YTDLP_ARGS="--cookies cookies.txt" or
# YTDLP_ARGS="--cookies-from-browser firefox" (see the yt-dlp FAQ on exporting YouTube cookies).
# Export them from a private/incognito window that is closed afterwards without logging out: cookies
# taken from a normal browser session are rotated by the browser within hours, and YouTube then
# rejects them mid-build ("account cookies are no longer valid"). yt-dlp rewrites the cookie file on
# every run. The check is tied to the egress IP and is intermittent: it can hit one cluster node and
# not another, and cookies do not always lift it, so running the download from a different machine
# (with or without cookies) is often the quickest fix. Only the download is host-sensitive.
# Alternatively download the two videos elsewhere and place them in SAVE_DIR as wmaze_src_v1.<ext>
# and wmaze_src_v2.<ext> (Video 1 must be the 300x240 rendition); present downloads are skipped.
# Every step is skipped when its output already exists and is complete (the frame count of every
# encoded video is checked: an interrupted ffmpeg run leaves a valid-looking but short file, which is
# deleted and redone), so the script can be re-run after an interruption (partial downloads are
# resumed). The videos are only deleted when the chunk counts come out as expected; otherwise they
# stay in place with a warning so the build can be checked. A finished build (config.json present,
# videos gone) is left untouched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
YTDLP_ARGS="${YTDLP_ARGS:-}"   # extra yt-dlp options, e.g. --cookies FILE (see the header comment)

# Paths are kept relative to datasets/ so that the recorded config.json matches the original
# ("windows_maze/windows_maze_20h_r64_fps20_train.mp4", save_dir "windows_maze", ...).
cd "$SCRIPT_DIR"
SAVE_DIR="${1:-windows_maze}"
mkdir -p "$SAVE_DIR"

CHUNK_SIZE=500        # frames per .npy chunk used for the paper's dataset
RESOLUTION=64
SEED=0                # the loader reads <save_dir>/{train,test}/<seed>/
T_TOTAL=1000000       # number of train frames exposed to the loader (first 1M frames)
TRAIN_MP4="$SAVE_DIR/windows_maze_20h_r64_fps20_train.mp4"
TEST_MP4="$SAVE_DIR/windows_maze_20h_r64_fps20_test.mp4"

# Frame counts of a complete build (both uploads are 36,000 s at 30 fps). An output within 1% of its
# count passes; an interrupted encode is far shorter and is redone.
N_SRC=1080000                        # 10 h x 30 fps
N_V1_TRIMMED=1079212                 # 35973.73 s x 30
N_JOINED=$((N_V1_TRIMMED + N_SRC))   # 2,159,212
N_FPS30=2159100                      # 71970 s x 30
N_FPS20=1439400                      # 71970 s x 20
N_TRAIN=1296002                      # 0-18:00:00 at 20 fps (the stream-copy cut lands on a keyframe: +2)
N_TEST=143419                        # 18:00:00-19:59:30

# A finished build has had its videos removed (see the end of the script); leave it alone rather
# than downloading and re-encoding everything again.
if [ -e "$SAVE_DIR/config.json" ] && [ ! -e "$TRAIN_MP4" ]; then
    echo "[skip] $SAVE_DIR/config.json exists and the videos were already removed; nothing to do"
    exit 0
fi

VIDEO1_ID="MHGnSqr9kK8"   # "10 Hours of Windows 3D Maze" (Dprotp)
VIDEO2_ID="Hs5pyyPTzDE"   # "10 Hours of Windows 3D Maze" (The Best Classic & Retro Screensavers)
# The recipe used "-S res:256" / "-S res:720" and got .webm (VP9) files back in 2024. Current yt-dlp
# prefers AV1 at the same resolution, which older ffmpeg builds (e.g. 4.2) cannot decode, so VP9 is
# asked for first. YouTube does not always serve it: Video 1 arrives as 300x240 VP9 (format 242),
# Video 2 currently as 1280x720 h264 in mkv. Either is fine; the recipe does not depend on the codec.
SORT1="res:256,vcodec:vp9"   # -> 300x240 (VP9 when offered)
SORT2="res:720,vcodec:vp9"   # -> 1280x720 (VP9 when offered, h264 otherwise)

for tool in yt-dlp ffmpeg ffprobe "$PYTHON"; do
    command -v "$tool" >/dev/null 2>&1 || { echo "error: '$tool' not found in PATH" >&2; exit 1; }
done

# Hand yt-dlp the deno binary explicitly: its own PATH search runs in Python, which does not expand a
# literal "~" in PATH entries, so a deno that the shell finds can still be invisible to yt-dlp.
if [[ "$YTDLP_ARGS" != *--js-runtimes* ]]; then
    if DENO_BIN="$(command -v deno 2>/dev/null)"; then
        YTDLP_ARGS="$YTDLP_ARGS --js-runtimes deno:$DENO_BIN"
    else
        echo "warning: no 'deno' in PATH; yt-dlp will warn and YouTube throttles the downloads to ~50 KB/s" >&2
    fi
fi

ff() { ffmpeg -nostdin -y -hide_banner "$@"; }   # -nostdin: a stray keystroke must not stop an encode

# complete HAVE EXPECTED : true when HAVE is a number within 1% below EXPECTED (or above it).
complete() { [ "${1:-x}" -eq "${1:-x}" ] 2>/dev/null && [ "$1" -ge $(( $2 * 99 / 100 )) ]; }

# encode OUTPUT EXPECTED_FRAMES CMD... : run CMD unless OUTPUT already holds ~EXPECTED_FRAMES frames.
# ffmpeg finalises the container when it is stopped (Ctrl-C, 'q'), so an interrupted encode looks like
# a finished file; only the frame count tells them apart. A short existing OUTPUT is deleted and redone,
# and a short fresh OUTPUT aborts the build.
encode() {
    local out="$1" expected="$2"; shift 2
    local have
    if [ -e "$out" ]; then
        have="$(nb_frames "$out" 2>/dev/null || true)"
        if complete "$have" "$expected"; then
            echo "[skip] $out exists ($have frames)"
            return 0
        fi
        echo "[redo] $out has ${have:-0} frames, expected ~$expected: incomplete encode, rebuilding it" >&2
        rm -f "$out"
    fi
    echo "[run ] $*"
    "$@"
    have="$(nb_frames "$out" 2>/dev/null || true)"
    complete "$have" "$expected" || {
        echo "error: $out has ${have:-0} frames, expected ~$expected (interrupted encode or short input)" >&2
        exit 1
    }
}

# download VIDEO_ID SORT_SPEC BASENAME : fetch a video with yt-dlp into $SAVE_DIR/BASENAME.<ext>
# and echo the resulting path. The container extension is whatever yt-dlp picks (webm/mp4/mkv).
# Partial downloads (*.part, *.ytdl) are not treated as finished; yt-dlp resumes them.
finished_download() { ls "$SAVE_DIR/$1".* 2>/dev/null | grep -v -e '\.part$' -e '\.ytdl$' | head -n 1 || true; }
download() {
    local id="$1" sort="$2" base="$3"
    local existing
    existing="$(finished_download "$base")"
    if [ -n "$existing" ]; then
        echo "[skip] $existing exists" >&2
    else
        echo "[run ] yt-dlp https://www.youtube.com/watch?v=$id -S $sort $YTDLP_ARGS" >&2
        # shellcheck disable=SC2086  # YTDLP_ARGS is intentionally word-split
        yt-dlp "https://www.youtube.com/watch?v=$id" -S "$sort" -o "$SAVE_DIR/$base.%(ext)s" $YTDLP_ARGS >&2 || {
            echo "error: yt-dlp could not download $id. Usual causes: an old yt-dlp release (YouTube rejects" >&2
            echo "       them; run yt-dlp -U) or YouTube's bot check (\"Sign in to confirm you're not a bot\")," >&2
            echo "       which needs browser cookies: YTDLP_ARGS=\"--cookies FILE\" or" >&2
            echo "       YTDLP_ARGS=\"--cookies-from-browser firefox\". See the header comment." >&2
            return 1
        }
        existing="$(finished_download "$base")"
        [ -n "$existing" ] || { echo "error: yt-dlp finished but $SAVE_DIR/$base.* is missing" >&2; return 1; }
    fi
    echo "$existing"
}

nb_frames() {
    ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 "$1"
}

dims() {  # prints WIDTHxHEIGHT of the first video stream
    ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$1"
}

duration() {  # prints the container duration in whole seconds
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" | cut -d. -f1
}

# check_download FILE : abort unless FILE is the full ~36,000 s upload (a partial or truncated download
# would otherwise be encoded as if it were complete).
check_download() {
    local d
    d="$(duration "$1" 2>/dev/null || true)"
    if ! [ "${d:-x}" -eq "${d:-x}" ] 2>/dev/null || [ "$d" -lt 35900 ]; then
        echo "error: $1 is ${d:-0} s long, expected ~36000 (incomplete download?); delete it and re-run" >&2
        exit 1
    fi
}

cd_save() { cd "$SCRIPT_DIR/$SAVE_DIR" 2>/dev/null || cd "$SAVE_DIR"; }

# ---------------------------------------------------------------------------------------------
# Video 1: 240p download -> crop the 240x240 centre of the 300x240 frame -> 64x64 @ 30fps -> trim
# ---------------------------------------------------------------------------------------------
SRC1="$(download "$VIDEO1_ID" "$SORT1" wmaze_src_v1)" || exit 1
# The crop offsets assume the 240p rendition of this 5:4 upload, which is 300x240 (this is
# what the original dataset was built from). Abort rather than silently crop something else.
SRC1_DIMS="$(dims "$SRC1" 2>/dev/null || true)"
if [ "$SRC1_DIMS" != "300x240" ]; then
    echo "error: expected the Video 1 download to be 300x240 but got ${SRC1_DIMS:-an unreadable file} ($SRC1);" >&2
    echo "       pick the 300x240 format explicitly (yt-dlp -F $VIDEO1_ID) before continuing." >&2
    exit 1
fi
check_download "$SRC1"
encode "$SAVE_DIR/windows_maze_10h_r64_v1.mp4" "$N_SRC" \
    ff -i "$SRC1" -filter:v "fps=30,crop=240:240:30:0,scale=64:64" \
        "$SAVE_DIR/windows_maze_10h_r64_v1.mp4"
encode "$SAVE_DIR/windows_maze_10h_r64_v1_trimmed.mp4" "$N_V1_TRIMMED" \
    ff -to 35973.73 -i "$SAVE_DIR/windows_maze_10h_r64_v1.mp4" \
        "$SAVE_DIR/windows_maze_10h_r64_v1_trimmed.mp4"

# ---------------------------------------------------------------------------------------------
# Video 2: 720p download -> 30fps -> resize the 16:9 frame to 300x240 -> same crop/scale as Video 1
# ---------------------------------------------------------------------------------------------
SRC2="$(download "$VIDEO2_ID" "$SORT2" wmaze_src_v2)" || exit 1
check_download "$SRC2"
encode "$SAVE_DIR/windows_maze_10h_r64_v2.mp4" "$N_SRC" \
    ff -i "$SRC2" -filter:v fps=30 "$SAVE_DIR/windows_maze_10h_r64_v2.mp4"
# -s 300x240 (not the docstring's -aspect 300:240): squashes the stretched 16:9 capture back to
# 5:4 so the centre crop below gives the same framing as Video 1. See the header comment.
encode "$SAVE_DIR/windows_maze_10h_r64_v2_edit.mp4" "$N_SRC" \
    ff -i "$SAVE_DIR/windows_maze_10h_r64_v2.mp4" -s 300x240 \
        "$SAVE_DIR/windows_maze_10h_r64_v2_edit.mp4"
encode "$SAVE_DIR/windows_maze_10h_r64_v2_final.mp4" "$N_SRC" \
    ff -i "$SAVE_DIR/windows_maze_10h_r64_v2_edit.mp4" \
        -filter:v "fps=30,crop=240:240:30:0,scale=64:64" "$SAVE_DIR/windows_maze_10h_r64_v2_final.mp4"

# ---------------------------------------------------------------------------------------------
# Join the two 10h clips (Video 1 first), trim to 71970 s, resample to 20 fps
# ---------------------------------------------------------------------------------------------
printf "file 'windows_maze_10h_r64_v1_trimmed.mp4'\nfile 'windows_maze_10h_r64_v2_final.mp4'\n" \
    > "$SAVE_DIR/concat_list.txt"
concat_join() { ( cd_save && ff -f concat -safe 0 -i concat_list.txt -c copy windows_maze_20h_r64.mp4 ); }
encode "$SAVE_DIR/windows_maze_20h_r64.mp4" "$N_JOINED" concat_join
# (the docstring calls this output windows_maze_20h_r64_final1.mp4 and then uses it as *_fps30.mp4)
encode "$SAVE_DIR/windows_maze_20h_r64_fps30.mp4" "$N_FPS30" \
    ff -ss 0.02 -to 71970.02 -i "$SAVE_DIR/windows_maze_20h_r64.mp4" \
        "$SAVE_DIR/windows_maze_20h_r64_fps30.mp4"
encode "$SAVE_DIR/windows_maze_20h_r64_fps20.mp4" "$N_FPS20" \
    ff -i "$SAVE_DIR/windows_maze_20h_r64_fps30.mp4" -filter:v "fps=20" \
        "$SAVE_DIR/windows_maze_20h_r64_fps20.mp4"

# ---------------------------------------------------------------------------------------------
# Split: first 18h -> train, 18:00:00-19:59:30 -> test (stream copy, so cuts land on keyframes)
# ---------------------------------------------------------------------------------------------
encode "$TRAIN_MP4" "$N_TRAIN" \
    ff -ss 00:00:00 -to 18:00:00 -i "$SAVE_DIR/windows_maze_20h_r64_fps20.mp4" -c copy "$TRAIN_MP4"
encode "$TEST_MP4" "$N_TEST" \
    ff -ss 18:00:00 -to 19:59:30 -i "$SAVE_DIR/windows_maze_20h_r64_fps20.mp4" -c copy "$TEST_MP4"

echo "train mp4 frames: $(nb_frames "$TRAIN_MP4")  (expected 1296002)"
echo "test  mp4 frames: $(nb_frames "$TEST_MP4")  (expected 143419)"

# ---------------------------------------------------------------------------------------------
# mp4 -> npy chunks. mp4_to_npy.py writes <save_dir>/train/<i>.npy, but the loader reads
# <save_dir>/train/<seed>/<i>.npy, so the chunks are moved into the seed subdirectory afterwards.
# ---------------------------------------------------------------------------------------------
N_TRAIN_CHUNKS=$((1296000 / CHUNK_SIZE))
N_TEST_CHUNKS=$((143000 / CHUNK_SIZE))
count_chunks() { ls "$SAVE_DIR/$1/$SEED" 2>/dev/null | grep -c '\.npy$' || true; }
if [ -e "$SAVE_DIR/config.json" ] && [ "$(count_chunks train)" -eq "$N_TRAIN_CHUNKS" ] \
        && [ "$(count_chunks test)" -eq "$N_TEST_CHUNKS" ]; then
    echo "[skip] $SAVE_DIR/config.json exists and the chunk counts are right"
else
    if [ -e "$SAVE_DIR/config.json" ]; then
        # config.json without the right chunk counts: chunks of an earlier, incomplete build
        echo "[redo] chunks in $SAVE_DIR are from an incomplete build (train $(count_chunks train)/$N_TRAIN_CHUNKS," \
             "test $(count_chunks test)/$N_TEST_CHUNKS); regenerating them" >&2
        rm -f "$SAVE_DIR"/train/*.npy "$SAVE_DIR"/test/*.npy \
              "$SAVE_DIR/train/$SEED"/*.npy "$SAVE_DIR/test/$SEED"/*.npy
    fi
    echo "[run ] mp4_to_npy.py"
    "$PYTHON" mp4_to_npy.py --train_video_path="$TRAIN_MP4" --test_video_path="$TEST_MP4" \
        --save_dir="$SAVE_DIR" --chunk_size="$CHUNK_SIZE" --resolution="$RESOLUTION" --seed="$SEED"
fi

for split in train test; do
    mkdir -p "$SAVE_DIR/$split/$SEED"
    if ls "$SAVE_DIR/$split"/*.npy >/dev/null 2>&1; then
        mv "$SAVE_DIR/$split"/*.npy "$SAVE_DIR/$split/$SEED/"
    fi
done

# mp4_to_npy.py records T_total as the number of frames decoded from the train mp4 (1,296,002).
# The dataset used for the paper caps it at 1,000,000 so that only the first 1M frames are
# used for training; the loader also asserts T_total % chunk_size == 0.
"$PYTHON" - "$SAVE_DIR/config.json" "$T_TOTAL" <<'EOF'
import json, sys
path, t_total = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(path))
cfg["T_total"] = t_total
json.dump(cfg, open(path, "w"), indent=2)
EOF

n_train=$(ls "$SAVE_DIR/train/$SEED" | grep -c '\.npy$' || true)
n_test=$(ls "$SAVE_DIR/test/$SEED" | grep -c '\.npy$' || true)
echo "train chunks: $n_train  (expected $((1296000 / CHUNK_SIZE)))"
echo "test  chunks: $n_test  (expected $((143000 / CHUNK_SIZE)))"
echo "config.json:"; cat "$SAVE_DIR/config.json"; echo

# ---------------------------------------------------------------------------------------------
# Remove the downloads and every intermediate video: the dataset is the chunks plus config.json.
# They are kept when the chunk counts are off, so that a failed build can be inspected.
# ---------------------------------------------------------------------------------------------
if [ "${KEEP_VIDEOS:-0}" = 1 ]; then
    echo "[skip] KEEP_VIDEOS=1: leaving the downloads and intermediate videos in $SAVE_DIR"
elif [ "$n_train" -ne $((1296000 / CHUNK_SIZE)) ] || [ "$n_test" -ne $((143000 / CHUNK_SIZE)) ]; then
    echo "warning: chunk counts differ from the expected values; leaving the videos in $SAVE_DIR so the" >&2
    echo "         build can be inspected (a truncated intermediate or a short source video is the usual cause)." >&2
else
    echo "[run ] rm downloads and intermediate videos in $SAVE_DIR"
    rm -f "$SAVE_DIR"/wmaze_src_v1.* "$SAVE_DIR"/wmaze_src_v2.* "$SAVE_DIR/concat_list.txt" \
        "$SAVE_DIR/windows_maze_10h_r64_v1.mp4" "$SAVE_DIR/windows_maze_10h_r64_v1_trimmed.mp4" \
        "$SAVE_DIR/windows_maze_10h_r64_v2.mp4" "$SAVE_DIR/windows_maze_10h_r64_v2_edit.mp4" \
        "$SAVE_DIR/windows_maze_10h_r64_v2_final.mp4" "$SAVE_DIR/windows_maze_20h_r64.mp4" \
        "$SAVE_DIR/windows_maze_20h_r64_fps30.mp4" "$SAVE_DIR/windows_maze_20h_r64_fps20.mp4" \
        "$TRAIN_MP4" "$TEST_MP4"
fi
echo "done: $SAVE_DIR"
