"""
Integration with a local install of llama.cpp
(https://github.com/ggml-org/llama.cpp).

TagGUI does not run any model inference itself. When captioning is started,
a `llama-server` process is launched with the user's GGUF model and
multimodal projector (mmproj) files, captions are generated through its
OpenAI-compatible `/v1/chat/completions` endpoint, and the server process is
terminated as soon as captioning is done to free VRAM.
"""
import base64
import io
import json
import re
import shlex
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

from PIL import Image as PilImage
from PIL.ImageOps import exif_transpose

from auto_captioning.model_families import get_chat_template
from utils.image import Image
from utils.settings import get_tag_separator

DEFAULT_PROMPT = 'Describe the image.'
HEALTH_ENDPOINT = '/health'
CHAT_COMPLETIONS_ENDPOINT = '/v1/chat/completions'
# The number of most recent `llama-server` output lines that are included in
# error messages.
SERVER_OUTPUT_TAIL_LINE_COUNT = 20


def replace_template_variable(match: re.Match, image: Image) -> str:
    template_variable = match.group(0)[1:-1].lower()
    if template_variable == 'tags':
        return ', '.join(image.tags)
    if template_variable == 'name':
        return image.path.stem
    if template_variable in ('directory', 'folder'):
        return image.path.parent.name
    return ''


def replace_template_variables(text: str, image: Image) -> str:
    # Replace template variables inside curly braces that are not escaped.
    text = re.sub(r'(?<!\\){[^{}]+(?<!\\)}',
                  lambda match: replace_template_variable(match, image), text)
    # Unescape escaped curly braces.
    text = re.sub(r'\\([{}])', r'\1', text)
    return text


def parse_bad_words(bad_words_string: str) -> list[str]:
    """
    Parse the comma-separated list of words and phrases that should be
    discouraged from the caption. Commas can be escaped with a backslash.
    """
    if not bad_words_string.strip():
        return []
    words = re.split(r'(?<!\\),', bad_words_string)
    words = [word.strip().replace(r'\,', ',') for word in words]
    return [word for word in words if word]


def strip_thinking_blocks(text: str) -> str:
    """
    Remove chain-of-thought blocks from the generated text. Recent
    `llama-server` builds parse thinking blocks into the separate
    `reasoning_content` response field, but this is a fallback for templates
    whose thinking tags are not recognized.
    """
    text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.DOTALL)
    # Remove a trailing thinking block that was never closed.
    text = re.sub(r'<think>.*$', ' ', text, flags=re.DOTALL)
    return text


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def encode_image_to_data_url(image: Image) -> str:
    """
    Load an image file, apply its EXIF orientation, and encode it as a
    base64 PNG data URL that can be sent to `llama-server`. Raises
    `PIL.UnidentifiedImageError` for corrupted or unsupported files.
    """
    pil_image = PilImage.open(image.path)
    pil_image = exif_transpose(pil_image)
    pil_image = pil_image.convert('RGB')
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    encoded_image = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded_image}'


class LlamaCppServerError(Exception):
    """An error related to starting or communicating with llama-server."""


class LlamaCppServer:
    """
    Manages the lifecycle of a `llama-server` process for one captioning run
    and generates captions through its HTTP API.
    """
    def __init__(self, caption_settings: dict, should_cancel=None):
        self.caption_settings = caption_settings
        self.should_cancel = should_cancel or (lambda: False)
        self.binary_path = caption_settings['llama_server_path']
        self.model_path = caption_settings['model_path']
        self.mmproj_path = caption_settings['mmproj_path']
        self.model_family = caption_settings['model_family']
        self.thinking_mode = caption_settings['thinking_mode']
        self.reasoning_effort = caption_settings['reasoning_effort']
        self.prompt = caption_settings['prompt']
        self.caption_start = caption_settings['caption_start']
        self.bad_words_string = caption_settings['bad_words']
        self.remove_tag_separators = caption_settings['remove_tag_separators']
        self.generation_parameters = caption_settings['generation_parameters']
        server_parameters = caption_settings['server_parameters']
        self.gpu_layer_count = server_parameters['gpu_layers']
        self.context_size = server_parameters['context_size']
        self.port_setting = server_parameters['port']
        self.startup_timeout = server_parameters['startup_timeout']
        self.generation_timeout = server_parameters['generation_timeout']
        self.additional_arguments = server_parameters['additional_arguments']

        self.process: subprocess.Popen | None = None
        self.output_reader_thread: threading.Thread | None = None
        # The most recent server output lines, kept for error messages.
        self.recent_server_output: deque[str] = deque(
            maxlen=SERVER_OUTPUT_TAIL_LINE_COUNT)
        self.port: int | None = None
        self.base_url: str | None = None

    def get_error_message(self) -> str | None:
        """
        Validate the caption settings before starting the server. Return an
        error message if something is misconfigured, and `None` otherwise.
        """
        if not self.binary_path.strip():
            return ('The path to the llama.cpp server binary is not set. Set '
                    'it in `File` → `Settings` → `llama.cpp server binary`, '
                    'for example '
                    '/home/user/AI/llama.cpp/build/bin/llama-server')
        binary_path = Path(self.binary_path.strip())
        if not binary_path.is_file():
            return (f'The llama.cpp server binary was not found at '
                    f'{binary_path}. Set the correct path in `File` → '
                    '`Settings`.')
        if not self.model_path.strip():
            return ('The path to the model GGUF file is not set. Set it in '
                    'the `Model (GGUF)` field of the Auto-Captioner.')
        model_path = Path(self.model_path.strip())
        if not model_path.is_file():
            return f'The model GGUF file was not found at {model_path}.'
        if model_path.suffix.lower() != '.gguf':
            return (f'The model file {model_path.name} is not a GGUF file. '
                    'llama.cpp requires models in the GGUF format.')
        if not self.mmproj_path.strip():
            return ('The path to the multimodal projector (mmproj) GGUF file '
                    'is not set. Vision models require an mmproj file, for '
                    'example mmproj-model-f16.gguf from the same release as '
                    'the model.')
        mmproj_path = Path(self.mmproj_path.strip())
        if not mmproj_path.is_file():
            return (f'The multimodal projector (mmproj) GGUF file was not '
                    f'found at {mmproj_path}.')
        if mmproj_path.suffix.lower() != '.gguf':
            return (f'The mmproj file {mmproj_path.name} is not a GGUF file.')
        return None

    def get_server_arguments(self) -> list[str]:
        arguments = [
            self.binary_path.strip(),
            '--model', self.model_path.strip(),
            '--mmproj', self.mmproj_path.strip(),
            '--host', '127.0.0.1',
            '--port', str(self.port),
            # Disable the built-in web interface.
            '--no-webui',
            # Use the Jinja template engine so that the chat template stored
            # in the GGUF metadata (or the one selected below) is applied.
            '--jinja',
            '--n-gpu-layers', str(self.gpu_layer_count),
            '--ctx-size', str(self.context_size),
        ]
        chat_template = get_chat_template(self.model_family)
        if chat_template:
            arguments += ['--chat-template', chat_template]
        # Thinking / reasoning control (for models that support it).
        if self.thinking_mode == 'Enabled':
            arguments += ['--reasoning', 'on']
        elif self.thinking_mode == 'Disabled':
            arguments += ['--reasoning', 'off']
        if (self.thinking_mode != 'Disabled'
                and self.reasoning_effort != 'Model default'):
            arguments += ['--reasoning-effort',
                          self.reasoning_effort.lower()]
        if self.additional_arguments.strip():
            arguments += shlex.split(self.additional_arguments)
        return arguments

    def read_process_output(self):
        """
        Continuously read the server output so that it is shown in the
        Auto-Captioner console and the process does not block on a full pipe
        buffer.
        """
        stdout = self.process.stdout
        if stdout is None:
            return
        try:
            for line in stdout:
                line = line.rstrip()
                self.recent_server_output.append(line)
                print(f'[llama-server] {line}')
        except ValueError:
            # The stream was closed while reading (during `stop()`).
            pass

    def start(self):
        """
        Start the llama-server process and wait until the model is loaded
        and the server is ready to accept requests.
        """
        self.port = (self.port_setting if self.port_setting > 0
                     else get_free_port())
        self.base_url = f'http://127.0.0.1:{self.port}'
        arguments = self.get_server_arguments()
        print('Starting the llama.cpp server...')
        print(' '.join(shlex.quote(argument) for argument in arguments))
        try:
            self.process = subprocess.Popen(
                arguments, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as error:
            raise LlamaCppServerError(
                f'Failed to start the llama.cpp server: {error}') from error
        self.output_reader_thread = threading.Thread(
            target=self.read_process_output, daemon=True)
        self.output_reader_thread.start()
        self.wait_until_ready()
        print('The llama.cpp server is ready.')

    def get_server_output_tail(self) -> str:
        # Give the output reader thread a moment to drain the remaining
        # output so that the error message is complete.
        if (self.output_reader_thread is not None
                and self.output_reader_thread.is_alive()):
            self.output_reader_thread.join(timeout=2)
        if not self.recent_server_output:
            return ''
        output = '\n'.join(self.recent_server_output)
        return f'\nThe last llama-server output lines:\n{output}'

    def wait_until_ready(self):
        """
        Poll the `/health` endpoint until the server reports that the model
        is loaded. Raises `LlamaCppServerError` on failure or timeout.
        """
        health_url = self.base_url + HEALTH_ENDPOINT
        start_time = time.monotonic()
        last_progress_message_time = start_time
        while True:
            if self.should_cancel():
                raise LlamaCppServerError('Captioning was canceled.')
            if self.process.poll() is not None:
                raise LlamaCppServerError(
                    f'The llama.cpp server exited unexpectedly with exit '
                    f'code {self.process.returncode}.'
                    f'{self.get_server_output_tail()}')
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > self.startup_timeout:
                raise LlamaCppServerError(
                    f'The llama.cpp server did not become ready within '
                    f'{self.startup_timeout} seconds. Loading large models '
                    'can take a while; the timeout can be increased in the '
                    f'advanced Auto-Captioner settings.'
                    f'{self.get_server_output_tail()}')
            try:
                request = urllib.request.Request(health_url)
                with urllib.request.urlopen(request, timeout=5) as response:
                    if response.status == 200:
                        return
            except urllib.error.HTTPError as error:
                # 503 means the model is still loading.
                if error.code != 503:
                    raise LlamaCppServerError(
                        f'The llama.cpp server returned an unexpected health '
                        f'check response (HTTP {error.code}).'
                        f'{self.get_server_output_tail()}') from error
            except urllib.error.URLError:
                # The server is not listening yet.
                pass
            if time.monotonic() - last_progress_message_time >= 15:
                last_progress_message_time = time.monotonic()
                print(f'Waiting for the llama.cpp server to load the model '
                      f'({elapsed_time:.0f} s)...')
            time.sleep(0.5)

    def get_image_prompt(self, image: Image) -> str:
        prompt = self.prompt.strip() or DEFAULT_PROMPT
        image_prompt = replace_template_variables(prompt, image)
        bad_words = parse_bad_words(self.bad_words_string)
        if bad_words:
            bad_words_string = ', '.join(f'"{word}"' for word in bad_words)
            image_prompt += (f'\n\nDo not use the following words or phrases '
                             f'in your answer: {bad_words_string}.')
        return image_prompt

    def get_request_messages(self, image: Image,
                             image_prompt: str) -> list[dict]:
        image_data_url = encode_image_to_data_url(image)
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url',
                 'image_url': {'url': image_data_url}},
                {'type': 'text', 'text': image_prompt}
            ]
        }]
        if self.caption_start.strip():
            # Prefill the assistant response so that the generated caption
            # continues this text.
            messages.append({'role': 'assistant',
                             'content': self.caption_start.strip()})
        return messages

    def get_request_payload(self, messages: list[dict]) -> dict:
        generation_parameters = self.generation_parameters
        payload = {
            'messages': messages,
            'stream': False,
            'max_tokens': generation_parameters['max_new_tokens'],
            'temperature': generation_parameters['temperature'],
            'top_p': generation_parameters['top_p'],
            'top_k': generation_parameters['top_k'],
            'min_p': generation_parameters['min_p'],
            'repeat_penalty': generation_parameters['repeat_penalty'],
        }
        return payload

    def post_chat_completion_request(self, payload: dict) -> dict:
        url = self.base_url + CHAT_COMPLETIONS_ENDPOINT
        request_data = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            url, data=request_data,
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(
                    request, timeout=self.generation_timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as error:
            error_body = ''
            try:
                error_body = error.read().decode('utf-8', errors='replace')
            except OSError:
                pass
            raise LlamaCppServerError(
                f'The llama.cpp server returned HTTP {error.code} for the '
                f'captioning request. Response: {error_body}') from error
        except urllib.error.URLError as error:
            raise LlamaCppServerError(
                f'Could not connect to the llama.cpp server: '
                f'{error.reason}') from error
        except TimeoutError as error:
            raise LlamaCppServerError(
                f'The captioning request timed out after '
                f'{self.generation_timeout} seconds.') from error

    def get_caption_from_response(self, response: dict,
                                  ) -> tuple[str, str | None]:
        """
        Extract the caption and an optional console output caption from a
        `/v1/chat/completions` response.
        """
        try:
            message = response['choices'][0]['message']
        except (KeyError, IndexError, TypeError) as error:
            raise LlamaCppServerError(
                f'Unexpected response from the llama.cpp server: '
                f'{json.dumps(response)[:500]}') from error
        content = message.get('content') or ''
        reasoning_content = message.get('reasoning_content') or ''
        generated_text = strip_thinking_blocks(content).strip()
        if not generated_text and reasoning_content:
            generated_text = strip_thinking_blocks(reasoning_content).strip()
        caption_start = self.caption_start.strip()
        if caption_start and generated_text.startswith(caption_start):
            # The server included the prefilled text in the response.
            caption = generated_text
        elif caption_start:
            caption = f'{caption_start} {generated_text}'
        else:
            caption = generated_text
        caption = caption.strip()
        if self.remove_tag_separators:
            caption = caption.replace(get_tag_separator(), ' ')
        console_output_caption = None
        if reasoning_content.strip():
            console_output_caption = (f'(thinking: '
                                      f'{reasoning_content.strip()[:300]})\n'
                                      f'{caption}')
        return caption, console_output_caption

    def generate_caption(self, image: Image,
                         image_prompt: str) -> tuple[str, str | None]:
        """
        Generate a caption for an image. Raises
        `PIL.UnidentifiedImageError` for unreadable image files and
        `LlamaCppServerError` for server or request errors.
        """
        messages = self.get_request_messages(image, image_prompt)
        payload = self.get_request_payload(messages)
        response = self.post_chat_completion_request(payload)
        return self.get_caption_from_response(response)

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        start_datetime_string = captioning_start_datetime.strftime(
            '%Y-%m-%d %H:%M:%S')
        model_name = Path(self.model_path).name
        if are_multiple_images_selected:
            return (f'Captioning with {model_name}... (start time: '
                    f'{start_datetime_string})')
        return f'Captioning with {model_name}...'

    def stop(self):
        """
        Terminate the llama-server process immediately to free VRAM and stop
        the output reader thread.
        """
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self.process.stdout is not None:
            try:
                self.process.stdout.close()
            except OSError:
                pass
        if self.output_reader_thread is not None:
            self.output_reader_thread.join(timeout=5)
            self.output_reader_thread = None
        self.process = None
