# TagGUI (llama.cpp edition)

<img src='images/icon.png' alt='TagGUI icon' width='128'>

A fork of [TagGUI](https://github.com/jhc13/taggui) for quickly adding and
editing image tags and captions, aimed towards creators of image datasets for
generative AI models.

This fork removes the built-in PyTorch/transformers inference stack. Instead,
all LLM inference for automatic captioning is outsourced to a separate local
install of [llama.cpp](https://github.com/ggml-org/llama.cpp):

- TagGUI starts a local `llama-server` process when captioning begins and
  terminates it immediately when captioning is done, so the VRAM is freed
  again.
- Any vision model supported by llama.cpp can be used, as long as it is
  available in the GGUF format (a model file plus its multimodal projector
  `mmproj` file).
- TagGUI passes the correct chat template to `llama-server` for the selected
  model family, and supports thinking/reasoning control where applicable.
- Models that llama.cpp does not support (such as the WD Tagger family) are
  no longer supported by this fork.

The setup is targeted at Linux with a CUDA-capable NVIDIA GPU.

<img src='images/screenshot-v1.25.0.png' alt='TagGUI screenshot' width='100%'>

## Features

- Keyboard-friendly interface for fast tagging
- Tag autocomplete based on your own most-used tags
- Integrated token counter (for Stable Diffusion)
- Automatic caption generation using local GGUF vision models via llama.cpp
- Batch tag operations for renaming, deleting, and sorting tags
- Advanced image list filtering

## Requirements

- Linux with a CUDA-capable GPU (the assumed setup for this fork)
- Python 3.11 or newer (Python 3.12 recommended)
- A local build of [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA
  support that includes the `llama-server` binary, for example built with:

  ```bash
  git clone https://github.com/ggml-org/llama.cpp
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON
  cmake --build build --config Release -j
  ```

  The binary is then located at
  `/path/to/llama.cpp/build/bin/llama-server`.
  A recent build is recommended so that all chat templates and the reasoning
  flags (`--reasoning`, `--reasoning-effort`) are available.

- A GGUF vision model and its multimodal projector (`mmproj`) GGUF file, for
  example from
  [the list of pre-quantized multimodal models in the llama.cpp
  documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).
  Popular choices include Qwen3-VL, Qwen2.5-VL, Gemma 3, InternVL, Llama 3.2
  Vision, MiniCPM-V, Mistral Small 3.1/3.2, Phi-4 Multimodal, SmolVLM, and
  Moondream2.

## Installation

Clone this repository and run the startup script:

```bash
git clone https://github.com/WelsCoding/taggui-lama
cd taggui-lama
./run.sh
```

`run.sh` handles all first-use startup tasks:

- If the virtual environment (`.venv`) does not exist yet, it creates it and
  installs the dependencies from `requirements.txt` into it.
- If the virtual environment already exists, it just starts the GUI.
- Run `./run.sh --reinstall` to recreate the virtual environment after
  dependency changes.

Alternatively, install the dependencies in `requirements.txt` manually into a
virtual environment and run `taggui/run_gui.py` with it.

## Usage

Load the directory containing your images by clicking the `Load Directory`
button in the center of the window (or `File` -> `Load Directory`).
Tags are loaded from `.txt` files in the directory with the same names as the
images.
Any changes you make to the tags are also automatically saved to these `.txt`
files.

## Automatic Captioning

Captioning works by talking to a local `llama-server` process. The server is
only started when you click `Start Auto-Captioning`, and it is terminated
immediately when captioning finishes (or is canceled), so the VRAM it uses is
freed again.

### Setup

1. Set the absolute path to the llama.cpp server binary in `File` ->
   `Settings` -> `llama.cpp server binary (llama-server)`, for example
   `/home/user/AI/llama.cpp/build/bin/llama-server`.
2. Optional: Set `Auto-captioning GGUF models directory` in `File` ->
   `Settings` to a directory containing your GGUF files. It is scanned
   recursively for `*.gguf` files, which then appear in the drop-down lists
   of the Auto-Captioner pane. Files whose names start with `mmproj` are
   treated as multimodal projector files (the llama.cpp naming convention).
3. In the Auto-Captioner pane, set:
   - `Model (GGUF)`: The path to the model file, for example
     `/models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf`.
   - `MMProj (GGUF)`: The path to the multimodal projector file belonging to
     the model, for example `/models/mmproj-Qwen3-VL-8B-F16.gguf`.
   - `Model family`: The model family, so that TagGUI can pass the correct
     chat template to `llama-server`. The default
     `Auto (template embedded in the GGUF)` uses the chat template stored in
     the GGUF metadata, which is correct for practically all modern GGUF
     releases. The other entries force one of the built-in llama.cpp chat
     templates (for example `chatml` for the Qwen VL families) and are meant
     as a fallback for GGUF files with a missing or broken embedded template.
4. Select the images you want to caption in the image list and click
   `Start Auto-Captioning`. The first request may take a while because the
   model has to be loaded into VRAM first; the loading progress of
   `llama-server` is shown in the Auto-Captioner console.

All vision models that llama.cpp supports work in TagGUI, since llama.cpp
handles the inference itself. TagGUI only passes the model and mmproj paths,
the chat template for the selected model family, and the generation
parameters to `llama-server`.

### Thinking / reasoning

For models that support thinking (chain-of-thought reasoning), such as the
Qwen3 family:

- `Thinking`: `Model default (auto)` lets the model decide, `Enabled` forces
  thinking on, and `Disabled` forces it off. This is passed to `llama-server`
  via its `--reasoning` argument.
- `Reasoning effort`: For templates that support it, the effort level
  (`Minimal`, `Low`, `Medium`, `High`, `xHigh`, `Max`) is passed via the
  `--reasoning-effort` argument of `llama-server`.

Thinking output is separated from the caption: the caption itself only
contains the final answer, while the reasoning trace (if any) is shown in the
console output for debugging.

### Captioning parameters

`Prompt`: Instructions given to the vision model. The chat template for the
selected model family is applied automatically by `llama-server`.
You can use the following template variables to dynamically insert information
about each image into the prompt:

- `{tags}`: The tags of the image, separated by commas.
- `{name}`: The file name of the image without the extension.
- `{directory}` or `{folder}`: The name of the directory containing the image.

An example prompt using a template variable could be
`Describe the image using the following tags as context: {tags}`.
With this prompt, `{tags}` would be replaced with the existing tags of each
image before the prompt is sent to the model.
If the prompt is empty, `Describe the image.` is used.

`Start caption with`: Generated captions will start with this text. The text
is also used to prefill the assistant response, so the model continues the
caption from it.

`Caption position`: Where the generated caption is inserted relative to the
existing tags of the image.

`Remove tag separators in caption`: If checked, tag separators (commas by
default) will be removed from the generated captions.

`Discourage from caption`: Words or phrases that should not be present in the
generated captions, separated by commas (`,`).
For example, you can put `appears,seems,possibly` to prevent the model from
using an uncertain tone in the captions.
Since llama.cpp does not support logit-level word banning, the words are
appended to the prompt as a negative instruction instead. The words may still
occasionally be generated.

The `Include in caption` option from the original TagGUI is not available in
this fork because it required beam search, which llama.cpp does not support.

### Advanced settings

The advanced settings in the Auto-Captioner pane map to `llama-server` and
its sampling parameters:

- `Maximum new tokens`: The maximum number of tokens to generate
  (`max_tokens` in the request).
- `Temperature`: `0` results in (mostly) deterministic, greedy generation.
- `Top-p`, `Top-k`, `Min-p`, `Repeat penalty`: Standard llama.cpp sampling
  parameters sent with each request.
- `GPU layers to offload`: Passed to `llama-server` as `--n-gpu-layers`
  (the default of `99` offloads the whole model to the GPU).
- `Context size (tokens)`: Passed as `--ctx-size`. Vision models consume a
  lot of context per image, so this should not be too small.
- `Server port (0 = automatic)`: The port `llama-server` listens on
  (localhost only). `0` picks a free port automatically.
- `Server startup timeout (s)`: How long to wait for the model to load.
- `Per-image timeout (s)`: How long a single captioning request may take.
- `Additional server arguments`: Extra command-line arguments appended to the
  `llama-server` invocation, for example `--flash-attn on`.

## Advanced Image List Filtering

The basic functionality of filtering for images that contain a certain tag is
available by clicking on the tag in the `All Tags` pane.
In addition to this, you can construct more complex filters in
the `Filter Images` box at the top of the `Images` pane.

<details>
<summary>
Click here to see the full documentation for the filter syntax.
</summary>

### Filter criteria

These are the prefixes you can use to specify the filter criteria you want to
apply:

- `tag:`: Images that have the filter term as a tag
    - `tag:cat` will match images with the tag `cat`.
- `caption`: Images that contain the filter term in the caption
    - The caption is the list of tags as a single string, as it appears in the
      `.txt` file.
    - `caption:cat` will match images that have `cat` anywhere in the
      caption. For example, images with the tag `orange cat` or the
      tag `catastrophe`.
- `name`: Images that contain the filter term in the file name
    - `name:cat` will match images such as `cat-1.jpg` or `large_cat.png`.
- `path`: Images that contain the filter term in the full file path
    - `path:cat` will match images such as `C:\Users\cats\dog.jpg` or
      `/home/dogs/cat.jpg`.
- You can also use a filter term with no prefix to filter for images that
  contain the term in either the caption or the file path.
    - `cat` will match images containing `cat` in the caption or file path.

The following are prefixes for numeric filters. The operators `=` (`==` also
works), `!=`, `<`, `>`, `<=`, and `>=` are used to specify the type of
comparison.

- `tags`: Images that have the specified number of tags
    - `tags:=13` will match images that have exactly 13 tags.
    - `tags:!=7` will match images that do not have exactly 7 tags (images with
      less than 7 tags or more than 7 tags).
- `chars`: Images that have the specified number of characters in the caption
    - `chars:<100` will match images that have less than 100 characters in the
      caption.
    - `chars:>=30` will match images that have 30 or more characters in the
      caption.
- `tokens`: Images that have the specified number of tokens in the caption
    - `tokens:>75` will match images that have more than 75 tokens in the
      caption.
    - `tokens:<=50` will match images that have 50 or fewer tokens in the
      caption.

### Spaces and quotes

If the filter term contains spaces, you must enclose it in quotes (either
single or double quotes).
For example, to find images with the tag `orange cat`, you must
use `tag:"orange cat"` or `tag:'orange cat'`.
If you have both spaces and quotes in the filter term, you can escape the
quotes with backslashes.
For example, you can use `tag:"orange \"cat\""` for the tag `orange "cat"`.
An alternative is to use different types of quotes for the outer and inner
quotes, like so: `tag:'orange "cat"'`.

### Wildcards

You can use the `*` character as a wildcard to match any number of any
characters, and the `?` character to match any single character.
For example, `tag:*cat` will match images with tags like `orange cat`,
`large cat`, and `cat`.

### Combining filters

Logical operators can be used to combine multiple filters:

- `NOT`: Images that do not match the filter
    - `NOT tag:cat` will match images that do not have the tag `cat`.
- `AND`: Images that match both filters before and after the operator
    - `tag:cat AND tag:orange` will match images that have both the tag `cat`
      and the tag `orange`.
- `OR`: Images that match either filter before or after the operator
    - `tag:cat OR tag:dog` will match images that have either the tag `cat` or
      the tag `dog`, or both.

The lowercase versions of these operators will also work: `not`, `and`,
and `or`.

The operator precedence is `NOT` > `AND` > `OR`, so by default, `NOT` will be
evaluated first, then `AND`, then `OR`.
You can use parentheses to change this order.
For example, in `tag:cat AND (tag:orange OR tag:white)`, the `OR` will be
evaluated first, matching images that have the tag `cat` and either the
tag `orange` or the tag `white`.
You can nest parentheses and operators to create arbitrarily complex filters.
</details>

## Controls

- ⭐ Previous / next image: `Ctrl`+`Up` / `Down` (just `Up` / `Down` also works
  in some cases)
- Jump to the first untagged image: `Ctrl`+`J`
- Focus the `Filter Images` box: `Alt`+`F`
- Focus the `Add Tag` box: `Alt`+`A`
- Focus the `Image Tags` list: `Alt`+`I`
- Focus the `Search Tags` box: `Alt`+`S`
- Focus the `Start Auto-Captioning` button: `Alt`+`C`

### Images pane

- First / last image: `Home` / `End`
- Select multiple images: Hold `Ctrl` or `Shift` and click the images
- Select all images: `Ctrl`+`A`
- Invert selection: `Ctrl`+`I`
- Right-clicking on an image will bring up the context menu, which includes
  actions such as copying and pasting tags and moving or copying selected
  images to another directory.

### Image Tags pane

- Add a tag: Type the tag into the `Add Tag` box and press `Enter`
- ⭐ Add the first tag suggested by autocomplete: `Ctrl`+`Enter`
- Add a tag to multiple images: Select the images in the image list add
  the tag
- Delete a tag: Select the tag and press `Delete`
- Rename a tag: Double-click the tag, or select the tag and press `F2`
- Reorder tags: Drag and drop the tags
- Select multiple tags: Hold `Ctrl` or `Shift` and click the tags

### All Tags pane

- Show all images containing a tag: Select the tag (When `Tag click action` is
  set to `Filter images for tag`)
- Add a tag to selected images: Click the tag (When `Tag click action` is set
  to `Add tag to selected images`)
- Delete all instances of a tag: Select the tag and press `Delete`
- Rename all instances of a tag: Double-click the tag, or select the tag and
  press `F2`

The `Edit` menu contains additional features for batch tag operations, such as
`Find and Replace` (`Ctrl`+`R`) and `Batch Reorder Tags` (`Ctrl`+`B`).
