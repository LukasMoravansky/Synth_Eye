"""
Synth.Eye - Vision-Based Industrial AI Application
Main UI Application

This application provides a user interface for camera-based industrial inspection.
Layout is optimized for 4K monitors with 1920×1200px input image resolution.
"""

import sys
import os
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer, QRectF, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QBrush, QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGraphicsView, QGraphicsScene, QSizePolicy
)

# ============================================================================
# Color Constants
# ============================================================================

# Primary Colors
COLOR_PRIMARY = "#533bff"  # Main brand color (buttons, logo, accents)

# Text Colors
COLOR_TEXT_DARK = "#333"    # Main text color
COLOR_TEXT_MEDIUM = "#666"  # Secondary text color
COLOR_TEXT_LIGHT = "#999"   # Tertiary text color

# Background Colors
COLOR_BG_WHITE = "#ffffff"      # White background
COLOR_BG_LIGHT = "#f0f0f0"      # Light gray background
COLOR_BG_DISABLED = "#e0e0e0"   # Disabled button background

# Border Colors
COLOR_BORDER_LIGHT = "#ddd"     # Light border
COLOR_BORDER_MEDIUM = "#ccc"    # Medium border
COLOR_BORDER_GRID = "#e0e0e0"   # Grid lines

# Graph Colors
COLOR_GRAPH_TOTAL = "#2ecc71"   # Green - Total line
COLOR_GRAPH_OK = "#27ae60"      # Dark green - OK line
COLOR_GRAPH_NOK = "#e74c3c"     # Red - NOK line

# Status Colors
COLOR_STATUS_SUCCESS_BG = "#e8f5e9"  # Light green background
COLOR_STATUS_SUCCESS_TEXT = "#2e7d32" # Green text
COLOR_STATUS_SUCCESS_BORDER = "#4caf50" # Green border

# ============================================================================
# Mock Camera Interface (for disabled camera option)
# ============================================================================

class MockCamera:
    """Mock camera class for testing UI without physical camera"""

    def __init__(self):
        self.connected = False
        self.resolution = None

    def connect(self):
        """Simulate camera connection"""
        self.connected = True
        self.resolution = (1920, 1200)
        return True

    def disconnect(self):
        """Simulate camera disconnection"""
        self.connected = False
        self.resolution = None

    def capture(self):
        """Simulate image capture - returns None for now"""
        if not self.connected:
            return None
        # Return a placeholder image (will be implemented later)
        return None

    def get_resolution(self):
        """Get camera resolution"""
        return self.resolution

# ============================================================================
# Aspect Ratio Label Widget
# ============================================================================

class AspectRatioLabel(QLabel):
    """QLabel that maintains a fixed aspect ratio"""

    def __init__(self, aspect_ratio=1.6, parent=None):  # 1920/1200 = 1.6
        super().__init__(parent)
        self.aspect_ratio = aspect_ratio
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self):
        """Return size hint that maintains aspect ratio"""
        width = self.width() if self.width() > 0 else 400
        height = int(width / self.aspect_ratio)
        return QSize(width, height)

    def resizeEvent(self, event):
        """Maintain aspect ratio on resize"""
        super().resizeEvent(event)
        width = event.size().width()
        height = int(width / self.aspect_ratio)
        if self.height() != height:
            self.setFixedHeight(height)

# ============================================================================
# Productivity Graph Widget
# ============================================================================

class ProductivityGraph(QWidget):
    """Line graph widget showing Total, OK, and NOK counts over iterations"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Use minimum size policy to allow responsive sizing
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {COLOR_BG_WHITE}; border: 1px solid {COLOR_BORDER_MEDIUM};")

        # Data storage
        self.iterations = []
        self.total_data = []
        self.ok_data = []
        self.nok_data = []

    def add_data_point(self, total, ok, nok):
        """Add a new data point to the graph"""
        iteration = len(self.iterations) + 1
        self.iterations.append(iteration)
        self.total_data.append(total)
        self.ok_data.append(ok)
        self.nok_data.append(nok)
        self.update()  # Trigger repaint
        self.repaint()  # Force immediate repaint

    def clear_data(self):
        """Clear all graph data"""
        self.iterations = []
        self.total_data = []
        self.ok_data = []
        self.nok_data = []
        self.update()

    def paintEvent(self, event):
        """Draw the line graph"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        
        # Increased margins for better label readability
        left_margin = 60  # More space for Y-axis labels and tick values
        right_margin = 30
        top_margin = 30
        bottom_margin = 50  # More space for X-axis label
        
        graph_width = width - left_margin - right_margin
        graph_height = height - top_margin - bottom_margin
        graph_x = left_margin
        graph_y = top_margin

        # Draw background
        painter.fillRect(self.rect(), QColor(COLOR_BG_WHITE))

        if not self.iterations:
            # Draw placeholder text
            painter.setPen(QColor(COLOR_TEXT_LIGHT))
            # Use responsive font size based on widget height
            font_size = max(12, int(self.height() * 0.04))
            painter.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, font_size))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data available")
            return

        # Find max value for scaling
        max_val = max(max(self.total_data) if self.total_data else 1,
                     max(self.ok_data) if self.ok_data else 1,
                     max(self.nok_data) if self.nok_data else 1, 1)

        # Draw axes
        painter.setPen(QPen(QColor(COLOR_TEXT_DARK), 2))
        # X-axis
        painter.drawLine(graph_x, height - bottom_margin, graph_x + graph_width, height - bottom_margin)
        # Y-axis
        painter.drawLine(graph_x, graph_y, graph_x, height - bottom_margin)

        # Draw axis labels with larger, more readable font
        axis_font_size = min(max(14, int(self.height() * 0.04)), 20)  # Limit max size to 28pt
        label_font = QFont(SynthEyeApp.EUROSTYLE_FONT, axis_font_size, QFont.Bold)
        painter.setFont(label_font)
        painter.setPen(QColor(COLOR_TEXT_DARK))
        
        # X-axis label (larger and more space)
        x_label_rect = QRectF(graph_x, height - bottom_margin + 10, graph_width, 30)
        painter.drawText(x_label_rect, Qt.AlignCenter, "Iteration")
        
        # Y-axis label (larger and more space)
        painter.save()
        painter.translate(15, height / 2)
        painter.rotate(-90)
        y_label_rect = QRectF(-60, 0, 120, 30)
        painter.drawText(y_label_rect, Qt.AlignCenter, "Count")
        painter.restore()

        # Draw grid lines
        painter.setPen(QPen(QColor(COLOR_BORDER_GRID), 1))
        for i in range(5):
            y = graph_y + (graph_height * i / 4)
            painter.drawLine(graph_x, y, graph_x + graph_width, y)

        # Draw data lines and points
        if len(self.iterations) >= 1:
            # Draw lines if we have more than one point
            if len(self.iterations) > 1:
                # OK line (darker green)
                painter.setPen(QPen(QColor(COLOR_GRAPH_OK), 2))
                for i in range(len(self.iterations) - 1):
                    x1 = graph_x + (graph_width * (self.iterations[i] - 1) / max(self.iterations))
                    y1 = height - bottom_margin - (graph_height * self.ok_data[i] / max_val)
                    x2 = graph_x + (graph_width * (self.iterations[i+1] - 1) / max(self.iterations))
                    y2 = height - bottom_margin - (graph_height * self.ok_data[i+1] / max_val)
                    painter.drawLine(x1, y1, x2, y2)

                # NOK line (red)
                painter.setPen(QPen(QColor(COLOR_GRAPH_NOK), 2))
                for i in range(len(self.iterations) - 1):
                    x1 = graph_x + (graph_width * (self.iterations[i] - 1) / max(self.iterations))
                    y1 = height - bottom_margin - (graph_height * self.nok_data[i] / max_val)
                    x2 = graph_x + (graph_width * (self.iterations[i+1] - 1) / max(self.iterations))
                    y2 = height - bottom_margin - (graph_height * self.nok_data[i+1] / max_val)
                    painter.drawLine(x1, y1, x2, y2)
            
            # Draw points for all data points (including single point)
            point_radius = 4
            # OK point (darker green)
            for i in range(len(self.iterations)):
                x = graph_x + (graph_width * (self.iterations[i] - 1) / max(self.iterations) if len(self.iterations) > 1 else 0)
                y = height - bottom_margin - (graph_height * self.ok_data[i] / max_val)
                painter.setPen(QPen(QColor(COLOR_GRAPH_OK), 2))
                painter.setBrush(QBrush(QColor(COLOR_GRAPH_OK)))
                painter.drawEllipse(x - point_radius, y - point_radius, point_radius * 2, point_radius * 2)
            
            # NOK point (red)
            for i in range(len(self.iterations)):
                x = graph_x + (graph_width * (self.iterations[i] - 1) / max(self.iterations) if len(self.iterations) > 1 else 0)
                y = height - bottom_margin - (graph_height * self.nok_data[i] / max_val)
                painter.setPen(QPen(QColor(COLOR_GRAPH_NOK), 2))
                painter.setBrush(QBrush(QColor(COLOR_GRAPH_NOK)))
                painter.drawEllipse(x - point_radius, y - point_radius, point_radius * 2, point_radius * 2)

        # Draw legend - responsive font size (if needed in future)
        legend_y = top_margin + 10
        legend_x = width - right_margin - 150
        legend_font_size = max(9, int(self.height() * 0.025))
        painter.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, legend_font_size))


# ============================================================================
# Main Application Window
# ============================================================================

class SynthEyeApp(QMainWindow):
    """Main application window for Synth.Eye"""

    EUROSTYLE_FONT = "Arial"  # Default, will be set in main()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synth.Eye - Vision-Based Industrial AI")

        # Camera state
        self.camera = MockCamera()
        self.captured_image = None
        self.analysis_result = None

        # Statistics
        self.total_scans = 0
        self.ok_count = 0
        self.nok_count = 0

        # Store responsive font sizes for use in dynamic updates
        self.title_font_size = None
        self.body_font_size = None

        # Setup UI
        self.init_ui()
        self.update_button_states()

    def init_ui(self):
        """Initialize the user interface"""
        # Set full-screen for 4K monitor
        self.showFullScreen()

        # Get screen size for responsive scaling
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # Calculate responsive sizes based on screen dimensions
        # Use ~1% of screen width/height for margins and spacing
        margin = max(10, int(screen_width * 0.01))
        spacing = max(10, int(screen_width * 0.01))

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setStyleSheet("background-color: white;")

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(spacing)
        main_layout.setContentsMargins(margin, margin, margin, margin)

        # Left side: Camera View and Controls
        left_layout = QVBoxLayout()
        left_layout.setSpacing(spacing)

        # Calculate responsive font sizes (base on screen height)
        title_font_size = max(16, int(screen_height * 0.025))
        body_font_size = max(12, int(screen_height * 0.010))
        button_font_size = max(14, int(screen_height * 0.010))

        # Store for use in dynamic updates
        self.title_font_size = title_font_size
        self.body_font_size = body_font_size

        # Camera View
        camera_label = QLabel("Camera View")
        camera_label.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, title_font_size, QFont.Bold))
        camera_label.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        left_layout.addWidget(camera_label)

        # Calculate camera view size based on screen width (smaller, ~40% of available width)
        # Maintain 1920×1200 aspect ratio (1.6:1)
        # camera_view_width = int(screen_width * 0.40)  # 40% of screen width
        camera_view_width = int(screen_width * 0.8)
        camera_view_height = int(camera_view_width / 1.6)  # Maintain 1920×1200 aspect ratio

        self.camera_view = AspectRatioLabel(aspect_ratio=1.6)  # 1920/1200 = 1.6
        self.camera_view.setText("Camera Feed")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet(f"""
            background-color: {COLOR_BG_LIGHT};
            color: {COLOR_TEXT_LIGHT};
            border: 2px solid {COLOR_BORDER_LIGHT};
            font-size: {body_font_size}pt;
        """)
        self.camera_view.setMaximumSize(camera_view_width, camera_view_height)
        left_layout.addWidget(self.camera_view)

        # Control Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(spacing)

        self.btn_connect = self.create_button("CONNECT", self.on_connect_clicked, COLOR_PRIMARY, button_font_size, screen_height)
        self.btn_capture = self.create_button("CAPTURE", self.on_capture_clicked, COLOR_PRIMARY, button_font_size, screen_height)
        self.btn_analyze = self.create_button("ANALYZE", self.on_analyze_clicked, COLOR_PRIMARY, button_font_size, screen_height)
        self.btn_clear = self.create_button("CLEAR", self.on_clear_clicked, COLOR_PRIMARY, button_font_size, screen_height)

        button_layout.addWidget(self.btn_connect)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_capture)
        button_layout.addWidget(self.btn_analyze)
        button_layout.addWidget(self.btn_clear)


        left_layout.addLayout(button_layout)

        left_layout.addStretch()

        # Bottom section: Logo and Status side by side
        bottom_layout = QHBoxLayout()
        bottom_layout.setAlignment(Qt.AlignBottom)

        # Branding section (left side)
        branding_layout = QVBoxLayout()
        branding_layout.setAlignment(Qt.AlignLeft | Qt.AlignBottom)

        # Logo image
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "Logo.png")
        logo_label = QLabel()
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale logo responsively (max 10% of screen height, maintain aspect ratio)
            max_logo_height = int(screen_height * 0.20)
            if pixmap.height() > max_logo_height:
                pixmap = pixmap.scaledToHeight(max_logo_height, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        branding_layout.addWidget(logo_label)

        bottom_layout.addLayout(branding_layout)
        # bottom_layout.addStretch()

        # Status section (right side)
        status_layout = QVBoxLayout()
        # status_layout.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        status_layout.setAlignment(Qt.AlignLeft| Qt.AlignTop)

        status_label = QLabel("Status")
        status_label.setAlignment(Qt.AlignLeft)
        status_label.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, title_font_size, QFont.Bold))
        status_label.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        status_layout.addWidget(status_label)

        self.status_camera = QLabel("Camera: Disconnected")
        self.status_camera.setAlignment(Qt.AlignLeft)
        self.status_camera.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, body_font_size))
        self.status_camera.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        status_layout.addWidget(self.status_camera)

        self.status_resolution = QLabel("Resolution: None")
        self.status_resolution.setAlignment(Qt.AlignLeft)
        self.status_resolution.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, body_font_size))
        self.status_resolution.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        status_layout.addWidget(self.status_resolution)

        bottom_layout.addLayout(status_layout)
        left_layout.addLayout(bottom_layout)

        # Right side: Logger and Graph
        right_layout = QVBoxLayout()
        right_layout.setSpacing(spacing)

        # System Logger
        logger_label = QLabel("System Logger")
        logger_label.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, title_font_size, QFont.Bold))
        logger_label.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        right_layout.addWidget(logger_label)

        self.logger = QTextEdit()
        self.logger.setReadOnly(True)
        self.logger.setStyleSheet(f"""
            background-color: {COLOR_BG_WHITE};
            color: {COLOR_TEXT_DARK};
            border: 2px solid {COLOR_BORDER_LIGHT};
            font-size: {body_font_size}pt;
            font-family: '{SynthEyeApp.EUROSTYLE_FONT}', monospace;
        """)
        right_layout.addWidget(self.logger, stretch=1)

        # Productivity Graph
        graph_label = QLabel("Productivity Graph")
        graph_label.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, title_font_size, QFont.Bold))
        graph_label.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        right_layout.addWidget(graph_label)

        self.productivity_graph = ProductivityGraph()
        right_layout.addWidget(self.productivity_graph, stretch=1)

        # Graph statistics text
        self.graph_stats = QLabel()
        self.graph_stats.setAlignment(Qt.AlignCenter)
        self.graph_stats.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, body_font_size))
        self.graph_stats.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        self.update_graph_stats()
        right_layout.addWidget(self.graph_stats)

        right_layout.addStretch()

        # Add layouts to main layout
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=1)

        # Log initial message
        self.log("Application started")

    def create_button(self, text, callback, color=COLOR_PRIMARY, font_size=18, screen_height=1080):
        """Create a styled button"""
        btn = QPushButton(text)
        # Calculate responsive button height (about 6% of screen height, min 50px)
        btn_height = max(50, int(screen_height * 0.06))
        # Calculate responsive padding
        padding_v = max(10, int(screen_height * 0.015))
        padding_h = max(20, int(screen_height * 0.02))

        btn.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, font_size, QFont.Bold))
        btn.setMinimumHeight(btn_height)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: {padding_v}px {padding_h}px;
                font-size: {font_size}pt;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
            QPushButton:disabled {{
                background-color: {COLOR_BG_DISABLED};
                color: {COLOR_TEXT_LIGHT};
                opacity: 0.1;
            }}
        """)
        btn.clicked.connect(callback)
        return btn

    def darken_color(self, color):
        """Darken a hex color for hover effect"""
        # Simple darkening - convert hex to RGB, reduce values
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        r, g, b = max(0, r - 30), max(0, g - 30), max(0, b - 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def update_button_states(self):
        """Update button enabled/disabled states based on camera connection"""
        if not self.camera.connected:
            # Disconnected: only CONNECT enabled
            self.btn_connect.setEnabled(True)
            self.btn_capture.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.btn_clear.setEnabled(False)
        else:
            # Connected: CONNECT becomes DISCONNECT, CAPTURE and CLEAR enabled
            self.btn_connect.setEnabled(True)
            self.btn_capture.setEnabled(True)
            self.btn_analyze.setEnabled(False)  # Only enabled after capture
            self.btn_clear.setEnabled(True)

    def on_connect_clicked(self):
        """Handle CONNECT/DISCONNECT button click"""
        if not self.camera.connected:
            # Connect
            if self.camera.connect():
                self.btn_connect.setText("DISCONNECT")
                self.log("Camera connected")
                self.update_status()
                self.update_button_states()
            else:
                self.log("Failed to connect to camera")
        else:
            # Disconnect
            self.camera.disconnect()
            self.btn_connect.setText("CONNECT")
            self.captured_image = None
            self.analysis_result = None
            self.camera_view.setText("Camera Feed")
            font_size = self.body_font_size if self.body_font_size else 16
            self.camera_view.setStyleSheet(f"""
                background-color: {COLOR_BG_LIGHT};
                color: {COLOR_TEXT_LIGHT};
                border: 2px solid {COLOR_BORDER_LIGHT};
                font-size: {font_size}pt;
            """)
            self.log("Camera disconnected")
            self.update_status()
            self.update_button_states()

    def on_capture_clicked(self):
        """Handle CAPTURE button click"""
        if not self.camera.connected:
            return

        self.log("Capture button pressed - performing scan")
        # Simulate capture (will be replaced with actual camera capture)
        self.captured_image = None  # Placeholder for actual image

        # For now, show a placeholder
        self.camera_view.setText("Image Captured")
        font_size = self.body_font_size if self.body_font_size else 16
        self.camera_view.setStyleSheet(f"""
            background-color: {COLOR_STATUS_SUCCESS_BG};
            color: {COLOR_STATUS_SUCCESS_TEXT};
            border: 2px solid {COLOR_STATUS_SUCCESS_BORDER};
            font-size: {font_size}pt;
        """)

        # Enable ANALYZE button
        self.btn_analyze.setEnabled(True)
        self.log("Image captured successfully")

    def on_analyze_clicked(self):
        """Handle ANALYZE button click"""
        # if self.captured_image is None:
        #     return

        self.log("Analyze button pressed - analyzing image")
        self.btn_analyze.setEnabled(False)

        # Simulate analysis with realistic results (90% OK rate, 10% NOK rate)
        import random
        # 90% chance of OK, 10% chance of NOK (realistic production scenario)
        result = "OK" if random.random() < 0.90 else "NOK"
        self.analysis_result = result

        # Update statistics
        self.total_scans += 1
        if result == "OK":
            self.ok_count += 1
        else:
            self.nok_count += 1

        self.log(f"Analysis result: {result}")

        # Update graph
        self.productivity_graph.add_data_point(
            self.total_scans, self.ok_count, self.nok_count
        )
        self.update_graph_stats()


    def on_clear_clicked(self):
        """Handle CLEAR button click"""
        self.log("Clear button pressed - clearing all data")

        # Clear graph data
        self.productivity_graph.clear_data()

        # Clear logger
        self.logger.clear()

        # Reset statistics
        self.total_scans = 0
        self.ok_count = 0
        self.nok_count = 0

        # Update display
        self.update_graph_stats()
        self.log("All data cleared")

    def update_status(self):
        """Update status display"""
        if self.camera.connected:
            resolution = self.camera.get_resolution()
            if resolution:
                self.status_camera.setText("Camera: Connected")
                self.status_resolution.setText(f"Resolution: {resolution[0]}x{resolution[1]}")
            else:
                self.status_camera.setText("Camera: Connected")
                self.status_resolution.setText("Resolution: None")
        else:
            self.status_camera.setText("Camera: Disconnected")
            self.status_resolution.setText("Resolution: None")

    def update_graph_stats(self):
        """Update the statistics text below the graph"""
        text = (
            f"The total number of images analyzed is <b>{self.total_scans}</b>, "
            f"of which <span style='color: green;'>{self.ok_count}</span> were found to be OK "
            f"and <span style='color: red;'>{self.nok_count}</span> were found to be NOK."
        )
        self.graph_stats.setText(text)

    def log(self, message):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.logger.append(f"{timestamp} – {message}")

# ============================================================================
# Font Loading
# ============================================================================

def load_eurostyle_font():
    """Load Eurostyle font from App/fonts directory"""
    font_dir = os.path.join(os.path.dirname(__file__), "fonts")

    # Try to load EuroStyle Normal.ttf first, then eurostile.TTF
    font_files = [
        os.path.join(font_dir, "EuroStyle Normal.ttf"),
        os.path.join(font_dir, "eurostile.TTF")
    ]

    font_family = None
    for font_file in font_files:
        if os.path.exists(font_file):
            font_id = QFontDatabase.addApplicationFont(font_file)
            if font_id != -1:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                if font_families:
                    font_family = font_families[0]
                    print(f"[INFO] Loaded font: {font_family} from {font_file}")
                    break

    if font_family is None:
        print("[WARNING] Could not load Eurostyle font, using default font")
        return "Arial"

    return font_family

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)

    # Load Eurostyle font
    eurostyle_font = load_eurostyle_font()

    # Store font name globally for use in the application
    SynthEyeApp.EUROSTYLE_FONT = eurostyle_font

    window = SynthEyeApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

