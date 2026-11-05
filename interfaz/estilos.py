ESTILO = '''
QMainWindow, QWidget { background: #081421; color: #e5edff; font-family: "Segoe UI", "DejaVu Sans"; font-size: 14px; }
QFrame#panel { background: #101f30; border: 1px solid #29445f; border-radius: 15px; }
QFrame#panel QLabel, QFrame#panel QWidget#transparente { background: transparent; }
QLabel#titulo { font-size: 26px; font-weight: 750; color: #d5e5ff; }
QLabel#seccion { font-size: 18px; font-weight: 700; }
QLabel#identidad { font-size: 30px; font-weight: 700; color: #d5e5ff; }
QLabel#metricas_voz { font-size: 13px; color: #a7bed9; }
QLabel#muted { color: #a7bed9; }
QLabel#estado { color: #39dfb1; font-weight: 600; }
QLabel#resultado { font-size: 30px; font-weight: 750; }
QLabel#comando { background: #10263b; border: 1px solid #2a4560; border-radius: 10px; padding: 7px; font-size: 24px; font-weight: 650; }
QPushButton { background: #10253a; border: 1px solid #355878; border-radius: 10px; padding: 6px 10px; font-weight: 600; }
QPushButton:hover { background: #193c5a; border-color: #30bfff; }
QPushButton:pressed { background: #075cba; }
QPushButton:checked { background: #0879f9; border-color: #35b8ff; }
QPushButton#principal { background: #087bff; border: 1px solid #35a9ff; font-size: 16px; padding: 8px 18px; }
QPushButton#principal:disabled, QPushButton:disabled { color: #718398; background: #142233; border-color: #263b51; }
QComboBox, QLineEdit { background: #0b1b2b; border: 1px solid #31526e; border-radius: 8px; padding: 8px; }
QComboBox QAbstractItemView { background: #10253a; selection-background-color: #087bff; }
QProgressBar { background: #07131f; border: none; border-radius: 3px; max-height: 5px; }
QProgressBar::chunk { background: #00c9f4; }
QToolTip { color: #ffffff; background: #18354d; border: 1px solid #3ca6e1; }
QSplitter::handle { background: #081421; width: 12px; }
'''
