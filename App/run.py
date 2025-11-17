import sys
import cv2
import random
from datetime import datetime
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QFont, QPen, QLinearGradient, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QTextEdit
)

# ---------------- Modern Gauge with Safe Animation ----------------
class ModernGauge(QWidget):
    """Radial gauge starting from top (12 o'clock), clockwise, with gaps, up to max_value"""
    def __init__(self, max_value=1000, color_start=QColor("#2ecc71"), color_end=QColor("#27ae60"), label="OK", parent=None):
        super().__init__(parent)
        self._value = 0
        self._target_value = 0
        self.max_value = max_value
        self.color_start = color_start
        self.color_end = color_end
        self.label = label
        self.setMinimumSize(100, 100)

        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_animation)

        self.gap_angle = 2.0  # degrees gap between each segment

    def set_value(self, value):
        if value > self.max_value:
            value = self.max_value
        self._target_value = value
        if not self.timer.isActive():
            self.timer.start()

    def update_animation(self):
        step = max(1, abs(self._target_value - self._value)//10)
        if self._value < self._target_value:
            self._value += step
        elif self._value > self._target_value:
            self._value -= step
        self.update()
        if self._value == self._target_value:
            self.timer.stop()

    def paintEvent(self, event):
        size = min(self.width(), self.height())
        margin = 20
        rect = QRectF(margin, margin, size - 2*margin, size - 2*margin)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        total_segments = self.max_value
        angle_per_segment = 360 / total_segments

        # Draw background segments
        for i in range(total_segments):
            start_angle = 90*16 - int(i * angle_per_segment * 16)  # start from top, clockwise
            span_angle = -int((angle_per_segment - self.gap_angle) * 16)  # negative = clockwise
            painter.setPen(QPen(QColor("#eee"), 2))
            painter.drawArc(rect, start_angle, span_angle)

        # Draw foreground segments for current value
        for i in range(self._value):
            start_angle = 90*16 - int(i * angle_per_segment * 16)
            span_angle = -int((angle_per_segment - self.gap_angle) * 16)
            gradient = QLinearGradient(0, 0, size, 0)
            gradient.setColorAt(0, self.color_start)
            gradient.setColorAt(1, self.color_end)
            painter.setPen(QPen(QBrush(gradient), 2))
            painter.drawArc(rect, start_angle, span_angle)

        # Value in center
        painter.setPen(QColor("#333"))
        painter.setFont(QFont("Arial", 36, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, f"{self._value}")

        # Label below
        painter.setFont(QFont("Arial", 18))
        painter.drawText(QRectF(0, size-100, self.width(), 20), Qt.AlignCenter, self.label)

# ---------------- Toggle Switch ----------------
class ToggleSwitch(QPushButton):
    """Custom Toggle Switch with sliding circle"""
    def __init__(self, width=150, height=40, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)
        self._bg_color_off = QColor("#e74c3c")
        self._bg_color_on = QColor("#2ecc71")
        self._circle_color = QColor("#ffffff")
        self._text_on = "Disconnect"
        self._text_off = "Connect"
        self._font = QFont("Arial", 14, QFont.Bold)
        self._circle_position = 2
        self.toggled.connect(self.animate)

        # Timer-based circle animation
        self.anim_timer = QTimer()
        self.anim_timer.setInterval(10)
        self.anim_timer.timeout.connect(self.update_circle)
        self._anim_target = 2
        self._anim_step = 0

    def set_texts(self, on_text: str, off_text: str):
        self._text_on = on_text
        self._text_off = off_text
        self.update()

    def set_font(self, font: QFont):
        self._font = font
        self.update()

    def animate(self, checked):
        start = 2 if checked else self.width() - self.height() + 2
        end = self.width() - self.height() + 2 if checked else 2
        self._anim_target = end
        distance = self._anim_target - self._circle_position
        self._anim_step = max(1, abs(distance)//5) * (1 if distance > 0 else -1)
        if not self.anim_timer.isActive():
            self.anim_timer.start()

    def update_circle(self):
        if self._circle_position == self._anim_target:
            self.anim_timer.stop()
            return
        if abs(self._circle_position - self._anim_target) <= abs(self._anim_step):
            self._circle_position = self._anim_target
        else:
            self._circle_position += self._anim_step
        self.update()

    def paintEvent(self, event):
        width = self.width()
        height = self.height()
        circle_radius = height - 4
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        rect_color = self._bg_color_on if self.isChecked() else self._bg_color_off
        painter.setBrush(QBrush(rect_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, width, height, height/2, height/2)

        # Circle
        painter.setBrush(QBrush(self._circle_color))
        painter.drawEllipse(self._circle_position, 2, circle_radius, circle_radius)

        # Text
        painter.setPen(Qt.white)
        painter.setFont(self._font)
        text = self._text_on if self.isChecked() else self._text_off
        painter.drawText(self.rect(), Qt.AlignCenter, text)
        painter.end()

# ---------------- Camera GUI ----------------
class CameraGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern Camera GUI")
        self.setMinimumSize(1350, 750)
        self.setStyleSheet("background-color: white;")

        self.analyzed_count = 0
        self.ok_count = 0
        self.nok_count = 0
        self.max_target = 50

        self.cap = None
        self.timer = None

        self.init_ui()

    def init_ui(self):
        # Video Feed
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color:#f0f0f0; color:black; border:1px solid #ccc;")
        self.video_label.setMinimumSize(600, 350)

        # Toggle
        self.toggle_camera_btn = ToggleSwitch()
        self.toggle_camera_btn.set_texts("Disconnect", "Connect")
        self.toggle_camera_btn.set_font(QFont("Arial", 14, QFont.Bold))
        self.toggle_camera_btn.toggled.connect(self.toggle_camera)

        # Action Buttons
        btn_height = 40
        self.btn_scan = self.create_button("Scan", self.scan, btn_height)
        self.btn_calibrate = self.create_button("Calibrate", self.calibrate, btn_height)
        self.btn_analyze = self.create_button("Analyze", self.analyze, btn_height)
        self.btn_clear = self.create_button("Clear", self.clear_data, btn_height)

        button_row = QHBoxLayout()
        button_row.setAlignment(Qt.AlignVCenter)
        button_row.addWidget(QLabel("Camera:"))
        button_row.addWidget(self.toggle_camera_btn)
        button_row.addStretch()
        for btn in [self.btn_scan, self.btn_calibrate, self.btn_analyze, self.btn_clear]:
            button_row.addWidget(btn)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label, stretch=5)
        left_layout.addLayout(button_row)
        left_layout.addStretch()

        # Info Box
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setStyleSheet("""
            background-color: #ffffff;
            color: #000000;
            border:1px solid #ccc;
            font-size:14px;
        """)

        # Gauges
        self.gauge_ok = ModernGauge(max_value=self.max_target, color_start=QColor("#2ecc71"), color_end=QColor("#27ae60"), label="OK")
        self.gauge_nok = ModernGauge(max_value=self.max_target, color_start=QColor("#e74c3c"), color_end=QColor("#c0392b"), label="NOK")
        gauge_layout = QHBoxLayout()
        gauge_layout.addWidget(self.gauge_ok)
        gauge_layout.addWidget(self.gauge_nok)

        # Modern info label under gauges
        self.gauge_info_label = QLabel()
        self.gauge_info_label.setAlignment(Qt.AlignCenter)
        self.gauge_info_label.setWordWrap(True)
        self.gauge_info_label.setFont(QFont("Arial", 18))
        self.gauge_info_label.setStyleSheet("color: #333;")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.info_box, stretch=2)
        right_layout.addLayout(gauge_layout, stretch=3)
        right_layout.addWidget(self.gauge_info_label)  # add info label

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=2)
        self.setLayout(main_layout)

        self.set_processing_buttons_enabled(False)

    # Button Creation
    def create_button(self, text, callback, height=40):
        btn = QPushButton(text)
        btn.setFixedHeight(height)
        btn.setStyleSheet("""
            QPushButton {
                background-color:#0078d4;
                color:white;
                padding:8px 15px;
                border-radius:6px;
                font-size:15px;
                border:1px solid #ccc;
            }
            QPushButton:hover {background-color:#005a9e;}
            QPushButton:disabled {background:#f0f0f0; color:#888;}
        """)
        btn.clicked.connect(callback)
        return btn

    # Camera Toggle
    def toggle_camera(self, state):
        if state:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.log("Failed to open webcam.")
                self.toggle_camera_btn.setChecked(False)
                return
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.log("Camera connected.")
            self.set_processing_buttons_enabled(True)
        else:
            if self.timer:
                self.timer.stop()
            if self.cap:
                self.cap.release()
            self.video_label.setText("Camera Feed")
            self.log("Camera disconnected.")
            self.set_processing_buttons_enabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label_width = self.video_label.width()
            label_height = self.video_label.height()
            img_height, img_width, _ = frame.shape
            scale = min(label_width/img_width, label_height/img_height)
            new_w, new_h = int(img_width*scale), int(img_height*scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            qimg = QImage(resized.data, new_w, new_h, new_w*3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            final_pixmap = QPixmap(label_width, label_height)
            final_pixmap.fill(Qt.white)
            painter = QPainter(final_pixmap)
            painter.drawPixmap((label_width-new_w)//2, (label_height-new_h)//2, pixmap)
            painter.end()
            self.video_label.setPixmap(final_pixmap)

    # Actions
    def scan(self): self.log("Scanning object...")
    def calibrate(self): self.log("Calibrating camera...")

    def analyze(self):
        self.log("Analyzing image...")
        self.set_processing_buttons_enabled(False)
        QTimer.singleShot(300, self.finish_analysis)

    def finish_analysis(self):
        try:
            result = random.choice(["OK", "NOK"])
            self.analyzed_count += 1
            if result=="OK":
                self.ok_count +=1
            else:
                self.nok_count +=1
            self.log(f"Result: {result}")
            self.update_gauges()
        except Exception as e:
            self.log(f"Error during analysis: {e}")
        finally:
            self.set_processing_buttons_enabled(True)

    def clear_data(self):
        self.analyzed_count = self.ok_count = self.nok_count = 0
        self.info_box.clear()
        self.update_gauges()
        self.log("Data cleared.")

    # Update Gauges
    def update_gauges(self):
        self.gauge_ok.set_value(self.ok_count)
        self.gauge_nok.set_value(self.nok_count)

        # Multi-line modern info text under gauges
        total = self.analyzed_count
        ok = self.ok_count
        nok = self.nok_count
        self.gauge_info_label.setText(
            f'The total number of images analyzed is <b>{total}</b>, '
            f'of which <span style="color:green">{ok}</span> were found to be OK '
            f'and <span style="color:red">{nok}</span> were found to be NOK.'
        )

    # Logging
    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.info_box.append(f"{timestamp} {message}")
        self.info_box.moveCursor(QTextCursor.End)

    def set_processing_buttons_enabled(self, enabled):
        for btn in [self.btn_scan, self.btn_calibrate, self.btn_analyze, self.btn_clear]:
            btn.setDisabled(not enabled)

# ---------------- Run App ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CameraGUI()
    gui.show()
    sys.exit(app.exec_())
