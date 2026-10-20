import sys
from pathlib import Path

from PySide6.QtCore import QModelIndex, Qt, Signal, Slot
from PySide6.QtGui import QFontMetrics, QTextCursor
from PySide6.QtWidgets import (QAbstractScrollArea, QDockWidget, QFileDialog,
                               QFormLayout, QFrame, QHBoxLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from auto_captioning.captioning_thread import CaptioningThread
from auto_captioning.model_families import (DEFAULT_MODEL_FAMILY,
                                            MODEL_FAMILY_LABELS)
from dialogs.caption_multiple_images_dialog import CaptionMultipleImagesDialog
from models.image_list_model import ImageListModel
from utils.big_widgets import TallPushButton
from utils.enums import CaptionPosition, ReasoningEffort, ThinkingMode
from utils.settings import DEFAULT_SETTINGS, get_settings, get_tag_separator
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsDoubleSpinBox,
                                    FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox,
                                    SettingsComboBox, SettingsLineEdit,
                                    SettingsPlainTextEdit)
from utils.utils import pluralize
from widgets.image_list import ImageList


def set_text_edit_height(text_edit: QPlainTextEdit, line_count: int):
    """
    Set the height of a text edit to the height of a given number of lines.
    """
    # From https://stackoverflow.com/a/46997337.
    document = text_edit.document()
    font_metrics = QFontMetrics(document.defaultFont())
    margins = text_edit.contentsMargins()
    height = int(font_metrics.lineSpacing() * line_count
                 + margins.top() + margins.bottom()
                 + document.documentMargin() * 2
                 + text_edit.frameWidth() * 2)
    text_edit.setFixedHeight(height)


class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Raised)


def get_local_gguf_paths(models_directory_path: Path,
                         mmproj_only: bool) -> list[str]:
    """
    Get the sorted paths of the GGUF files under a directory. By llama.cpp
    convention, multimodal projector files start with `mmproj`. Depending on
    `mmproj_only`, only the mmproj files or only the non-mmproj files are
    returned.
    """
    gguf_paths = models_directory_path.glob('**/*.gguf')
    matching_paths = []
    for gguf_path in gguf_paths:
        is_mmproj = gguf_path.name.lower().startswith('mmproj')
        if is_mmproj == mmproj_only:
            matching_paths.append(str(gguf_path))
    matching_paths.sort()
    return matching_paths


class PathComboBox(QWidget):
    """
    An editable combo box for a file path, persisted to the settings, with a
    browse button. Paths found in the GGUF models directory are added to the
    combo box, and a path can also be entered or browsed for manually.
    """
    def __init__(self, key: str, browse_caption: str, known_paths: list[str]):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.combo_box = SettingsComboBox(key=key, default='')
        self.combo_box.setEditable(True)
        # `setEditable()` must be called before `addItems()` to preserve any
        # path that was set previously.
        self.combo_box.addItems(known_paths)
        if not get_settings().value(key, '', type=str):
            # Show the placeholder text instead of the first known path when
            # no path has been set yet.
            self.combo_box.setCurrentText('')
        self.browse_button = QPushButton('Browse...')
        self.browse_button.setFixedWidth(
            int(self.browse_button.sizeHint().width() * 1.1))
        self.browse_button.clicked.connect(
            lambda: self.browse_for_path(browse_caption))
        layout.addWidget(self.combo_box, stretch=1)
        layout.addWidget(self.browse_button)

    @Slot()
    def browse_for_path(self, caption: str):
        initial_path = self.combo_box.currentText().strip()
        initial_directory = str(Path(initial_path).parent) if initial_path \
            else ''
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self, caption=caption, dir=initial_directory,
            filter='GGUF files (*.gguf);;All files (*)')
        if file_path:
            self.combo_box.setCurrentText(file_path)

    def currentText(self) -> str:
        return self.combo_box.currentText()


class CaptionSettingsForm(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        local_model_paths, local_mmproj_paths = self.get_local_gguf_paths()

        basic_settings_form = QFormLayout()
        basic_settings_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        basic_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.model_path_combo_box = PathComboBox(
            key='llama_model_path',
            browse_caption='Select the model GGUF file',
            known_paths=local_model_paths)
        self.model_path_combo_box.combo_box.lineEdit().setPlaceholderText(
            '/path/to/model.gguf')
        self.mmproj_path_combo_box = PathComboBox(
            key='llama_mmproj_path',
            browse_caption='Select the multimodal projector (mmproj) GGUF '
                           'file',
            known_paths=local_mmproj_paths)
        self.mmproj_path_combo_box.combo_box.lineEdit().setPlaceholderText(
            '/path/to/mmproj-model-f16.gguf')
        self.model_family_combo_box = FocusedScrollSettingsComboBox(
            key='llama_model_family', default=DEFAULT_MODEL_FAMILY)
        self.model_family_combo_box.addItems(MODEL_FAMILY_LABELS)
        self.prompt_text_edit = SettingsPlainTextEdit(key='prompt')
        set_text_edit_height(self.prompt_text_edit, 4)
        self.caption_start_line_edit = SettingsLineEdit(key='caption_start')
        self.caption_start_line_edit.setClearButtonEnabled(True)
        self.caption_position_combo_box = FocusedScrollSettingsComboBox(
            key='caption_position')
        self.caption_position_combo_box.addItems(list(CaptionPosition))
        self.remove_tag_separators_container = QWidget()
        remove_tag_separators_layout = QHBoxLayout(
            self.remove_tag_separators_container)
        remove_tag_separators_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        remove_tag_separators_layout.setContentsMargins(0, 0, 0, 0)
        self.remove_tag_separators_check_box = SettingsBigCheckBox(
            key='remove_tag_separators', default=True)
        remove_tag_separators_label = QLabel(
            'Remove tag separators in caption')
        remove_tag_separators_layout.addWidget(remove_tag_separators_label)
        remove_tag_separators_layout.addWidget(
            self.remove_tag_separators_check_box)
        self.thinking_mode_combo_box = FocusedScrollSettingsComboBox(
            key='llama_thinking_mode', default=ThinkingMode.AUTO.value)
        self.thinking_mode_combo_box.addItems([mode.value for mode
                                               in ThinkingMode])
        self.reasoning_effort_combo_box = FocusedScrollSettingsComboBox(
            key='llama_reasoning_effort',
            default=ReasoningEffort.DEFAULT.value)
        self.reasoning_effort_combo_box.addItems([effort.value for effort
                                                  in ReasoningEffort])

        basic_settings_form.addRow('Model (GGUF)', self.model_path_combo_box)
        basic_settings_form.addRow('MMProj (GGUF)',
                                   self.mmproj_path_combo_box)
        basic_settings_form.addRow('Model family',
                                   self.model_family_combo_box)
        self.prompt_label = QLabel('Prompt')
        basic_settings_form.addRow(self.prompt_label, self.prompt_text_edit)
        self.caption_start_label = QLabel('Start caption with')
        basic_settings_form.addRow(self.caption_start_label,
                                   self.caption_start_line_edit)
        basic_settings_form.addRow('Caption position',
                                   self.caption_position_combo_box)
        basic_settings_form.addRow(self.remove_tag_separators_container)
        basic_settings_form.addRow('Thinking',
                                   self.thinking_mode_combo_box)
        basic_settings_form.addRow('Reasoning effort',
                                   self.reasoning_effort_combo_box)

        self.toggle_advanced_settings_form_button = TallPushButton(
            'Show Advanced Settings')

        self.advanced_settings_form_container = QWidget()
        advanced_settings_form = QFormLayout(
            self.advanced_settings_form_container)
        advanced_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        advanced_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.bad_words_line_edit = SettingsLineEdit(key='bad_words')
        self.bad_words_line_edit.setClearButtonEnabled(True)
        advanced_settings_form.addRow('Discourage from caption',
                                      self.bad_words_line_edit)
        self.max_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='max_new_tokens', default=256, minimum=1, maximum=32768)
        self.temperature_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='temperature', default=0, minimum=0, maximum=2)
        self.temperature_spin_box.setSingleStep(0.05)
        self.top_p_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='top_p', default=1, minimum=0, maximum=1)
        self.top_p_spin_box.setSingleStep(0.01)
        self.top_k_spin_box = FocusedScrollSettingsSpinBox(
            key='top_k', default=40, minimum=0, maximum=1000)
        self.min_p_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='min_p', default=0.05, minimum=0, maximum=1)
        self.min_p_spin_box.setSingleStep(0.01)
        self.repeat_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='repeat_penalty', default=1.1, minimum=0, maximum=2)
        self.repeat_penalty_spin_box.setSingleStep(0.05)
        advanced_settings_form.addRow('Maximum new tokens',
                                      self.max_new_token_count_spin_box)
        advanced_settings_form.addRow('Temperature',
                                      self.temperature_spin_box)
        advanced_settings_form.addRow('Top-p', self.top_p_spin_box)
        advanced_settings_form.addRow('Top-k', self.top_k_spin_box)
        advanced_settings_form.addRow('Min-p', self.min_p_spin_box)
        advanced_settings_form.addRow('Repeat penalty',
                                      self.repeat_penalty_spin_box)
        advanced_settings_form.addRow(HorizontalLine())
        # The key is versioned ('_v2') because the meaning of the setting
        # changed: the old default of 99 forced all layers into VRAM and
        # disabled llama.cpp's automatic memory fitting, which caused CUDA
        # out-of-memory errors for models larger than the VRAM. The new
        # default of -1 leaves the choice to llama-server.
        self.gpu_layer_count_spin_box = FocusedScrollSettingsSpinBox(
            key='llama_gpu_layers_v2', default=-1, minimum=-1, maximum=999)
        self.context_size_spin_box = FocusedScrollSettingsSpinBox(
            key='llama_context_size', default=8192, minimum=1024,
            maximum=1048576)
        self.context_size_spin_box.setSingleStep(1024)
        self.server_port_spin_box = FocusedScrollSettingsSpinBox(
            key='llama_server_port', default=0, minimum=0, maximum=65535)
        self.startup_timeout_spin_box = FocusedScrollSettingsSpinBox(
            key='llama_server_startup_timeout', default=300, minimum=30,
            maximum=3600)
        self.startup_timeout_spin_box.setSingleStep(30)
        self.generation_timeout_spin_box = FocusedScrollSettingsSpinBox(
            key='llama_generation_timeout', default=600, minimum=30,
            maximum=7200)
        self.generation_timeout_spin_box.setSingleStep(30)
        self.additional_server_arguments_line_edit = SettingsLineEdit(
            key='llama_server_additional_arguments')
        self.additional_server_arguments_line_edit.setClearButtonEnabled(
            True)
        advanced_settings_form.addRow('GPU layers to offload (-1 = auto)',
                                      self.gpu_layer_count_spin_box)
        advanced_settings_form.addRow('Context size (tokens)',
                                      self.context_size_spin_box)
        advanced_settings_form.addRow('Server port (0 = automatic)',
                                      self.server_port_spin_box)
        advanced_settings_form.addRow('Server startup timeout (s)',
                                      self.startup_timeout_spin_box)
        advanced_settings_form.addRow('Per-image timeout (s)',
                                      self.generation_timeout_spin_box)
        advanced_settings_form.addRow(
            'Additional server arguments',
            self.additional_server_arguments_line_edit)
        self.advanced_settings_form_container.hide()

        self.addLayout(basic_settings_form)
        self.horizontal_line = HorizontalLine()
        self.addWidget(self.horizontal_line)
        self.addWidget(self.toggle_advanced_settings_form_button)
        self.addWidget(self.advanced_settings_form_container)
        self.addStretch()

        self.thinking_mode_combo_box.currentTextChanged.connect(
            self.set_reasoning_effort_enabled)
        self.toggle_advanced_settings_form_button.clicked.connect(
            self.toggle_advanced_settings_form)

        self.set_reasoning_effort_enabled(
            self.thinking_mode_combo_box.currentText())

    def get_local_gguf_paths(self) -> tuple[list[str], list[str]]:
        """
        Get the local model and mmproj GGUF paths found under the GGUF models
        directory set in the settings.
        """
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        if not models_directory_path:
            return [], []
        models_directory_path = Path(models_directory_path)
        if not models_directory_path.is_dir():
            return [], []
        print(f'Loading local GGUF paths under {models_directory_path}...')
        model_paths = get_local_gguf_paths(models_directory_path,
                                           mmproj_only=False)
        mmproj_paths = get_local_gguf_paths(models_directory_path,
                                            mmproj_only=True)
        total_path_count = len(model_paths) + len(mmproj_paths)
        print(f'Loaded {len(model_paths)} model and {len(mmproj_paths)} '
              f'mmproj {pluralize("path", total_path_count)}.')
        return model_paths, mmproj_paths

    @Slot(str)
    def set_reasoning_effort_enabled(self, thinking_mode: str):
        # The reasoning effort has no effect when thinking is disabled.
        self.reasoning_effort_combo_box.setEnabled(
            thinking_mode != ThinkingMode.DISABLED.value)

    @Slot()
    def toggle_advanced_settings_form(self):
        if self.advanced_settings_form_container.isHidden():
            self.advanced_settings_form_container.show()
            self.toggle_advanced_settings_form_button.setText(
                'Hide Advanced Settings')
        else:
            self.advanced_settings_form_container.hide()
            self.toggle_advanced_settings_form_button.setText(
                'Show Advanced Settings')

    def get_caption_settings(self) -> dict:
        llama_server_path = self.settings.value(
            'llama_server_path',
            defaultValue=DEFAULT_SETTINGS['llama_server_path'], type=str)
        return {
            'llama_server_path': llama_server_path,
            'model_path': self.model_path_combo_box.currentText(),
            'mmproj_path': self.mmproj_path_combo_box.currentText(),
            'model_family': self.model_family_combo_box.currentText(),
            'thinking_mode': self.thinking_mode_combo_box.currentText(),
            'reasoning_effort': self.reasoning_effort_combo_box.currentText(),
            'prompt': self.prompt_text_edit.toPlainText(),
            'caption_start': self.caption_start_line_edit.text(),
            'caption_position': self.caption_position_combo_box.currentText(),
            'remove_tag_separators':
                self.remove_tag_separators_check_box.isChecked(),
            'bad_words': self.bad_words_line_edit.text(),
            'generation_parameters': {
                'max_new_tokens': self.max_new_token_count_spin_box.value(),
                'temperature': self.temperature_spin_box.value(),
                'top_p': self.top_p_spin_box.value(),
                'top_k': self.top_k_spin_box.value(),
                'min_p': self.min_p_spin_box.value(),
                'repeat_penalty': self.repeat_penalty_spin_box.value()
            },
            'server_parameters': {
                'gpu_layers': self.gpu_layer_count_spin_box.value(),
                'context_size': self.context_size_spin_box.value(),
                'port': self.server_port_spin_box.value(),
                'startup_timeout': self.startup_timeout_spin_box.value(),
                'generation_timeout': self.generation_timeout_spin_box.value(),
                'additional_arguments':
                    self.additional_server_arguments_line_edit.text()
            }
        }


@Slot()
def restore_stdout_and_stderr():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class AutoCaptioner(QDockWidget):
    caption_generated = Signal(QModelIndex, str, list)

    def __init__(self, image_list_model: ImageListModel,
                 image_list: ImageList):
        super().__init__()
        self.image_list_model = image_list_model
        self.image_list = image_list
        self.settings = get_settings()
        self.is_captioning = False
        self.captioning_thread = None
        # Whether the last block of text in the console text edit should be
        # replaced with the next block of text that is outputted.
        self.replace_last_console_text_edit_block = False

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('auto_captioner')
        self.setWindowTitle('Auto-Captioner')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.start_cancel_button = TallPushButton('Start Auto-Captioning')
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat('%v / %m images captioned (%p%)')
        self.progress_bar.hide()
        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 4)
        self.console_text_edit.setReadOnly(True)
        self.console_text_edit.hide()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.start_cancel_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.console_text_edit)
        self.caption_settings_form = CaptionSettingsForm()
        layout.addLayout(self.caption_settings_form)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)
        self.setWidget(scroll_area)

        self.start_cancel_button.clicked.connect(
            self.start_or_cancel_captioning)

    @Slot()
    def start_or_cancel_captioning(self):
        if self.is_captioning:
            # Cancel captioning.
            self.captioning_thread.is_canceled = True
            self.start_cancel_button.setEnabled(False)
            self.start_cancel_button.setText('Canceling Auto-Captioning...')
        else:
            # Start captioning.
            self.generate_captions()

    def set_is_captioning(self, is_captioning: bool):
        self.is_captioning = is_captioning
        button_text = ('Cancel Auto-Captioning' if is_captioning
                       else 'Start Auto-Captioning')
        self.start_cancel_button.setText(button_text)

    @Slot(str)
    def update_console_text_edit(self, text: str):
        # '\x1b[A' is the ANSI escape sequence for moving the cursor up.
        if text == '\x1b[A':
            self.replace_last_console_text_edit_block = True
            return
        text = text.strip()
        if not text:
            return
        if self.console_text_edit.isHidden():
            self.console_text_edit.show()
        if self.replace_last_console_text_edit_block:
            self.replace_last_console_text_edit_block = False
            # Select and remove the last block of text.
            self.console_text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.console_text_edit.moveCursor(
                QTextCursor.MoveOperation.StartOfBlock,
                QTextCursor.MoveMode.KeepAnchor)
            self.console_text_edit.textCursor().removeSelectedText()
            # Delete the newline.
            self.console_text_edit.textCursor().deletePreviousChar()
        self.console_text_edit.appendPlainText(text)

    @Slot()
    def show_alert(self):
        if self.captioning_thread.is_canceled:
            return
        if self.captioning_thread.is_error:
            icon = QMessageBox.Icon.Critical
            text = ('An error occurred during captioning. See the '
                    'Auto-Captioner console for more information.')
        else:
            icon = QMessageBox.Icon.Information
            text = 'Captioning has finished.'
        alert = QMessageBox()
        alert.setIcon(icon)
        alert.setText(text)
        alert.exec()

    @Slot()
    def generate_captions(self):
        selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        show_alert_when_finished = False
        if selected_image_count > 1:
            confirmation_dialog = CaptionMultipleImagesDialog(
                selected_image_count)
            reply = confirmation_dialog.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return
            show_alert_when_finished = (confirmation_dialog
                                        .show_alert_check_box.isChecked())
        self.set_is_captioning(True)
        caption_settings = self.caption_settings_form.get_caption_settings()
        if caption_settings['caption_position'] != CaptionPosition.DO_NOT_ADD:
            self.image_list_model.add_to_undo_stack(
                action_name=f'Generate '
                            f'{pluralize("Caption", selected_image_count)}',
                should_ask_for_confirmation=selected_image_count > 1)
        if selected_image_count > 1:
            self.progress_bar.setRange(0, selected_image_count)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
        tag_separator = get_tag_separator()
        self.captioning_thread = CaptioningThread(
            self, self.image_list_model, selected_image_indices,
            caption_settings, tag_separator)
        self.captioning_thread.text_outputted.connect(
            self.update_console_text_edit)
        self.captioning_thread.clear_console_text_edit_requested.connect(
            self.console_text_edit.clear)
        self.captioning_thread.caption_generated.connect(
            self.caption_generated)
        self.captioning_thread.progress_bar_update_requested.connect(
            self.progress_bar.setValue)
        self.captioning_thread.finished.connect(
            lambda: self.set_is_captioning(False))
        self.captioning_thread.finished.connect(restore_stdout_and_stderr)
        self.captioning_thread.finished.connect(self.progress_bar.hide)
        self.captioning_thread.finished.connect(
            lambda: self.start_cancel_button.setEnabled(True))
        if show_alert_when_finished:
            self.captioning_thread.finished.connect(self.show_alert)
        # Redirect `stdout` and `stderr` so that the outputs are displayed in
        # the console text edit.
        sys.stdout = self.captioning_thread
        sys.stderr = self.captioning_thread
        self.captioning_thread.start()
