package main.java.gui;

import java.awt.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.prefs.Preferences;

import javax.swing.BorderFactory;
import javax.swing.*;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.Scrollable;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;
import javax.swing.filechooser.FileNameExtensionFilter;

import main.java.input.CModification;
import org.apache.commons.lang3.StringUtils;

import com.formdev.flatlaf.FlatDarkLaf;
import com.formdev.flatlaf.FlatLaf;
import com.formdev.flatlaf.FlatLightLaf;

import ai.djl.util.cuda.CudaUtils;
import main.java.input.CParameter;
import main.java.util.GPUTools;
import main.java.util.GenericUtils;
import main.java.util.PyInstaller;
import org.apache.tools.ant.types.Commandline;

public class CarafeGUI extends JFrame {

    // global workflow selection
    public static int globalWorkflowIndex = 0;
    private static String carafe_library_directory = "";

    // Brand colors only (keep header/action identity; let FlatLaf handle general UI
    // colors)
    private static final Color PRIMARY_COLOR = new Color(41, 128, 185);
    private static final Color PRIMARY_DARK = new Color(31, 97, 141);
    private static final Color PRIMARY_LIGHT = new Color(52, 152, 219);
    private static final Color ACCENT_COLOR = new Color(46, 204, 113);

    // Layout spacing constants
    private static final int ROW_SPACING = 4; // Vertical spacing between rows
    private static final int COL_SPACING = 8; // Horizontal spacing between columns
    private static final Insets DEFAULT_INSETS = new Insets(ROW_SPACING, COL_SPACING, ROW_SPACING, COL_SPACING);

    // Window size constants
    private static final int DEFAULT_WIDTH = 700; // Default window width
    private static final int DEFAULT_HEIGHT = 750; // Default window height
    private static final int MIN_WIDTH = 700; // Minimum window width
    private static final int MIN_HEIGHT = 750; // Minimum window height
    private static final int COMPONENT_HEIGHT = 32; // Standard height for input components
    private boolean enforcingMinSize = false;

    // Input fields
    private JComboBox<String> workflowCombo;
    private JTextField diannReportFileField;
    private JTextField trainMsFileField;
    private JTextField trainDbFileField;
    private JTextField projectMsFileField;
    private JTextField libraryDbFileField;
    private JTextField outputDirField;
    private JComboBox<String> pythonPathCombo;
    private JComboBox<String> diannPathCombo;
    private JComboBox<String> msConvertPathCombo;
    private JTextField carafeAdditionalOptionsField;
    private JTextField diannAdditionalOptionsField;

    // Multi-file selection storage
    private java.util.List<String> trainMsFiles = new java.util.ArrayList<>();
    private java.util.List<String> projectMsFiles = new java.util.ArrayList<>();

    // Input panel rows components for dynamic visibility
    private java.util.List<JComponent> diannReportRowComponents;
    private java.util.List<JComponent> trainMsRowComponents;
    private java.util.List<JComponent> trainDbRowComponents;
    private java.util.List<JComponent> projectMsRowComponents;
    private java.util.List<JComponent> diannAdditionalOptionsRowComponents;
    private java.util.List<JComponent> libraryDbRowComponents;
    private java.util.List<JComponent> diannExeRowComponents;
    private java.util.List<JComponent> msConvertExeRowComponents;
    private JPanel inputFieldsPanel;

    // Training Data Generation settings
    private JSpinner fdrSpinner;
    private JSpinner ptmSiteProbSpinner;
    private JSpinner ptmSiteQvalueSpinner;
    private JSpinner fragTolSpinner;
    private JComboBox<String> fragTolUnitCombo;
    private JCheckBox refineBoundaryCheckbox;
    private JTextField rtPeakWindowField;
    private JSpinner xicCorSpinner;
    private JSpinner minFragMzSpinner;
    private JSpinner nIonMinSpinner;
    private JSpinner cIonMinSpinner;

    // Model Training settings
    private JComboBox<String> modeCombo;
    private JTextField nceField;
    private JComboBox<String> msInstrumentField;
    private JComboBox<String> deviceCombo;

    // Library Generation settings
    private JComboBox<String> enzymeCombo;
    private JSpinner missCleavageSpinner;
    private JComboBox<String> fixModAvailableCombo;
    private JTextField fixModSelectedField;
    private JComboBox<String> varModAvailableCombo;
    private JTextField varModSelectedField;
    private JSpinner maxVarSpinner;
    private JCheckBox clipNmCheckbox;
    private JSpinner minLengthSpinner;
    private JSpinner maxLengthSpinner;
    private JSpinner minPepMzSpinner;
    private JSpinner maxPepMzSpinner;
    private JSpinner minPepChargeSpinner;
    private JSpinner maxPepChargeSpinner;
    private JSpinner libMinFragMzSpinner;
    private JSpinner libMaxFragMzSpinner;
    private JSpinner LibTopNFragIonsSpinner;
    private JSpinner libMinNumFragSpinner;
    private JSpinner libFragNumMinSpinner;
    private JComboBox<String> libraryFormatCombo;

    // Output console
    private JTextArea consoleArea;
    private JProgressBar progressBar;
    private JButton runButton;
    private JButton stopButton;
    private JTabbedPane tabbedPane;
    private JLabel statusLabel;
    private JScrollPane consoleScrollPane;
    private JScrollPane inputScrollPane;

    // Header/theme refs
    private JPanel headerPanel;
    private JPanel headerTitlePanel;
    private JPanel headerTextPanel;
    private JPanel headerRightPanel;
    private JLabel headerIconLabel;
    private JLabel headerTitleLabel;
    private JLabel headerSubtitleLabel;
    private JLabel headerVersionLabel;
    private JToggleButton darkModeToggle;

    // Track created info cards so we can re-theme them on toggle
    private final java.util.List<InfoCardRef> infoCards = new java.util.ArrayList<>();

    // Debounce timer for MSConvert visibility updates
    private javax.swing.Timer msConvertVisibilityDebounceTimer;

    // Execution
    private ExecutorService executor;
    private Process currentProcess;
    private volatile boolean isRunning = false;
    private String cachedGpuStatus = "Checking..."; // Field to hold the result

    // Preferences for remembering last used directory
    private static final Preferences prefs = Preferences.userNodeForPackage(CarafeGUI.class);
    private static final String PREF_LAST_DIR = "lastDirectory";
    private static final String PREF_PYTHON_PATH = "pythonPath";
    private static final String PREF_DIANN_PATH = "diannPath";
    private static final String PREF_MSCONVERT_PATH = "msConvertPath";
    private static final String PREF_DARK_MODE = "darkMode";

    /**
     * Time usage tracking map.
     */
    private final java.util.Map<String, Double> timeUsageMap = new java.util.LinkedHashMap<>();

    private String diannVersion = "";
    private boolean isDiannV2 = false;

    private BufferedWriter logWriter;

    public CarafeGUI() {
        setTitle("Carafe - Spectral Library Generator");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setMinimumSize(new Dimension(MIN_WIDTH, MIN_HEIGHT));
        setPreferredSize(new Dimension(DEFAULT_WIDTH, DEFAULT_HEIGHT));
        setResizable(true);

        // Hide icon in title bar (requires FlatLaf window decorations)
        getRootPane().putClientProperty("JRootPane.titleBarShowIcon", false);

        // Load Application Icon (for Taskbar)
        try {
            java.net.URL iconUrl = getClass().getResource("/carafe-icon.png");
            if (iconUrl != null) {
                ImageIcon icon = new ImageIcon(iconUrl);
                setIconImage(icon.getImage());
                if (java.awt.Taskbar.isTaskbarSupported()
                        && java.awt.Taskbar.getTaskbar().isSupported(java.awt.Taskbar.Feature.ICON_IMAGE)) {
                    java.awt.Taskbar.getTaskbar().setIconImage(icon.getImage());
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to load application icon: " + e.getMessage());
        }

        // Load persisted theme preference
        boolean dark = prefs.getBoolean(PREF_DARK_MODE, false);

        // Set look and feel
        try {
            if (dark) {
                FlatDarkLaf.setup();
            } else {
                FlatLightLaf.setup();
            }
            customizeUIDefaults(); // A) global UI polish
        } catch (Exception e) {
            e.printStackTrace();
        }

        initComponents();
        applyThemeToCustomComponents(); // Important: sync custom-colored components to current theme
        updateGpuStatusAsync(); // Initial check

        pack();

        // Dynamic sizing based on monitor resolution
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        int newWidth;
        int newHeight;
        if (screenSize.height > 1080) {
            int targetWidth = DEFAULT_WIDTH + 100;
            int targetHeight = DEFAULT_HEIGHT + 100;

            // Ensure we respect minimums and don't exceed screen
            newWidth = Math.max(MIN_WIDTH, Math.min(targetWidth, screenSize.width));
            newHeight = Math.max(MIN_HEIGHT, Math.min(targetHeight, screenSize.height));
        } else {
            Dimension packedSize = getSize();
            Dimension minSize = getMinimumSize();
            newWidth = Math.max(packedSize.width, minSize.width);
            newHeight = Math.max(packedSize.height, minSize.height);
        }
        setSize(newWidth, newHeight);
        setLocationRelativeTo(null);
    }

    /**
     * Centralized UI Defaults.
     * Can now be made static since it only touches UIManager.
     */
    private static void customizeUIDefaults() {
        UIManager.put("defaultFont", new Font("Segoe UI", Font.PLAIN, 13));
        UIManager.put("Button.arc", 10);
        UIManager.put("Component.arc", 10);
        UIManager.put("ProgressBar.arc", 10);
        UIManager.put("TextComponent.arc", 8);
        UIManager.put("TabbedPane.showTabSeparators", true);
        UIManager.put("TabbedPane.tabInsets", new Insets(8, 14, 8, 14));
        UIManager.put("ScrollBar.width", 12);
        UIManager.put("ScrollBar.thumbArc", 999);
        UIManager.put("ScrollBar.thumbInsets", new Insets(2, 2, 2, 2));
        UIManager.put("Component.focusWidth", 1);
        UIManager.put("Component.innerFocusWidth", 0);
        UIManager.put("Component.hideMnemonics", true);
        ToolTipManager.sharedInstance().setDismissDelay(30000);
    }

    private static Color lafColor(String key, Color fallback) {
        Color c = UIManager.getColor(key);
        return c != null ? c : fallback;
    }

    private synchronized void logToConsole(String message) {
        SwingUtilities.invokeLater(() -> {
            consoleArea.append(message);
            consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
        });

        if (logWriter != null) {
            try {
                logWriter.write(message);
                logWriter.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void initComponents() {
        setLayout(new BorderLayout());

        // Header
        add(createHeader(), BorderLayout.NORTH);

        // Main content with tabs
        tabbedPane = new JTabbedPane();
        tabbedPane.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
        tabbedPane.setFont(new Font("Segoe UI", Font.PLAIN, 13));

        inputScrollPane = wrapInScrollPane(createInputPanel());
        tabbedPane.addTab("Workflow", inputScrollPane);
        tabbedPane.addTab("Training Data Generation", wrapInScrollPane(createTrainingDataPanel()));
        tabbedPane.addTab("Model Training", wrapInScrollPane(createModelTrainingPanel()));
        tabbedPane.addTab("Library Generation", wrapInScrollPane(createLibraryGenerationPanel()));
        tabbedPane.addTab("Console", createConsolePanel());

        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 15, 10, 15));
        mainPanel.add(tabbedPane, BorderLayout.CENTER);

        add(mainPanel, BorderLayout.CENTER);

        // Footer with run button
        add(createFooter(), BorderLayout.SOUTH);

        // Ensure scroll pane starts at top
        SwingUtilities.invokeLater(() -> {
            if (inputScrollPane != null && inputScrollPane.getViewport() != null) {
                inputScrollPane.getViewport().setViewPosition(new Point(0, 0));
            }
        });
    }

    private JScrollPane wrapInScrollPane(JPanel panel) {
        JScrollPane scrollPane = new JScrollPane(panel);
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        scrollPane.setBorder(null);
        scrollPane.getVerticalScrollBar().setUnitIncrement(16);
        scrollPane.getHorizontalScrollBar().setUnitIncrement(16);
        stripScrollPaneBorder(scrollPane);
        return scrollPane;
    }

    /**
     * Helper to robustly remove borders from ScrollPanes to ensure clean UI.
     * Can accept either the ScrollPane itself or a specific inner component.
     */
    private void stripScrollPaneBorder(JComponent c) {
        if (c == null)
            return;

        // 1. If component is itself a JScrollPane
        if (c instanceof JScrollPane sp) {
            sp.setBorder(BorderFactory.createEmptyBorder());
            sp.setViewportBorder(BorderFactory.createEmptyBorder());
        }

        // 2. Traversal check (Nuclear option for updates)
        SwingUtilities.invokeLater(() -> {
            JScrollPane sp = (c instanceof JScrollPane) ? (JScrollPane) c
                    : (JScrollPane) SwingUtilities.getAncestorOfClass(JScrollPane.class, c);
            if (sp != null) {
                sp.setBorder(BorderFactory.createEmptyBorder());
                sp.setViewportBorder(BorderFactory.createEmptyBorder());
            }
        });
    }

    // Generic helper for header toggle buttons
    private JToggleButton createHeaderToggleButton(String initialText, boolean initialSelected,
            java.awt.event.ActionListener action) {
        JToggleButton btn = new JToggleButton(initialText) {
            @Override
            protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                try {
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                    boolean dark = FlatLaf.isLafDark();
                    Color bg = dark ? new Color(0, 0, 0, 60) : new Color(255, 255, 255, 60);
                    if (getModel().isRollover())
                        bg = withAlpha(bg, 100);

                    g2.setColor(bg);
                    g2.fillRoundRect(0, 0, getWidth(), getHeight(), 15, 15);

                    g2.setColor(withAlpha(getForeground(), 80));
                    g2.drawRoundRect(0, 0, getWidth() - 1, getHeight() - 1, 15, 15);
                } finally {
                    g2.dispose();
                }
                super.paintComponent(g);
            }
        };
        btn.setSelected(initialSelected);
        btn.setContentAreaFilled(false);
        btn.setBorderPainted(false);
        btn.setOpaque(false);
        btn.setFocusPainted(false);
        btn.setMargin(new Insets(6, 16, 6, 16));
        btn.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        btn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        btn.setPreferredSize(new Dimension(100, 30));
        btn.addActionListener(action);
        return btn;
    }

    // Dynamic Header Panel Class
    private class DynamicHeaderPanel extends JPanel {
        private final java.util.List<Particle> particles = new java.util.ArrayList<>();
        private final javax.swing.Timer timer;
        private static final int INITIAL_PARTICLE_COUNT = 80;
        private static final double CONNECTION_THRESHOLD = 130.0;
        // Pre-computed Color lookup table to avoid object churn in animation loop
        private static final Color[] WHITE_ALPHA = new Color[256];
        static {
            for (int i = 0; i < 256; i++) {
                WHITE_ALPHA[i] = new Color(255, 255, 255, i);
            }
        }
        private boolean animationEnabled = true;
        private int lastWidth = 0;
        private int lastHeight = 0;

        DynamicHeaderPanel() {
            super(new BorderLayout());
            // Initialize particles
            for (int i = 0; i < INITIAL_PARTICLE_COUNT; i++) {
                particles.add(new Particle());
            }

            // Animation loop
            timer = new javax.swing.Timer(33, e -> {
                if (!animationEnabled)
                    return;
                int w = getStyleableWidth();
                int h = getStyleableHeight();
                for (Particle p : particles) {
                    p.update(w, h);
                }
                repaint();
            });
            timer.start();

            // Pause animation when not visible to save CPU
            addComponentListener(new java.awt.event.ComponentAdapter() {
                @Override
                public void componentShown(java.awt.event.ComponentEvent e) {
                    if (animationEnabled)
                        timer.start();
                }

                @Override
                public void componentHidden(java.awt.event.ComponentEvent e) {
                    timer.stop();
                }
            });

            // Pause when window is minimized
            addHierarchyListener(e -> {
                if ((e.getChangeFlags() & java.awt.event.HierarchyEvent.SHOWING_CHANGED) != 0) {
                    if (isShowing() && animationEnabled) {
                        timer.start();
                    } else {
                        timer.stop();
                    }
                }
            });
        }

        void setAnimationEnabled(boolean enabled) {
            this.animationEnabled = enabled;
            if (enabled) {
                timer.start();
            } else {
                timer.stop();
            }
            repaint();
        }

        private int getStyleableWidth() {
            return getWidth() > 0 ? getWidth() : 800;
        }

        private int getStyleableHeight() {
            return getHeight() > 0 ? getHeight() : 150;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            int w = getWidth();
            int h = getHeight();

            if (w <= 0 || h <= 0)
                return;

            // Handle initialization or resize
            boolean initialized = (lastWidth != 0);
            boolean resized = (Math.abs(w - lastWidth) > 50 || Math.abs(h - lastHeight) > 50);

            if (!initialized) {
                // First draw: distribute everything
                for (Particle p : particles) {
                    p.reset(true);
                }
                lastWidth = w;
                lastHeight = h;
            } else if (resized) {
                // Resize event
                if (w > lastWidth) {
                    // Expanded: scatter some particles to fill new space
                    // Move ~40% of particles to new random locations
                    for (int i = 0; i < particles.size(); i++) {
                        if (i % 5 <= 1) { // roughly 40%
                            particles.get(i).reset(true);
                        }
                    }
                } else {
                    // Shrunk: bring in outliers immediately
                    for (Particle p : particles) {
                        if (p.x > w)
                            p.x = Math.random() * w;
                        if (p.y > h)
                            p.y = Math.random() * h;
                    }
                }
                lastWidth = w;
                lastHeight = h;
            }

            boolean dark = FlatLaf.isLafDark();
            Color base = lafColor("Carafe.headerBase", new Color(0x2F82B7));

            Color top = dark ? adjust(base, -40) : adjust(base, 60);
            Color mid = dark ? adjust(base, -20) : adjust(base, 35);
            Color bottom = dark ? adjust(base, -10) : adjust(base, 15);

            Graphics2D g2 = (Graphics2D) g.create();
            try {
                g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                LinearGradientPaint paint = new LinearGradientPaint(
                        0f, 0f, 0f, (float) h,
                        new float[] { 0f, 0.5f, 1f },
                        new Color[] { top, mid, bottom });
                g2.setPaint(paint);
                g2.fillRect(0, 0, w, h);

                if (animationEnabled) {
                    drawParticles(g2, w, h);
                }

                int highlightH = (int) (h * 0.5f);
                Color hiTop = new Color(255, 255, 255, dark ? 15 : 30);
                Color hiBot = new Color(255, 255, 255, 0);
                g2.setPaint(new GradientPaint(0, 0, hiTop, 0, highlightH, hiBot));
                g2.fillRect(0, 0, w, highlightH);
            } finally {
                g2.dispose();
            }
        }

        private void drawParticles(Graphics2D g2, int w, int h) {
            double thresholdSq = CONNECTION_THRESHOLD * CONNECTION_THRESHOLD;

            g2.setStroke(new BasicStroke(1.0f));
            for (int i = 0; i < particles.size(); i++) {
                Particle p1 = particles.get(i);
                for (int j = i + 1; j < particles.size(); j++) {
                    Particle p2 = particles.get(j);
                    double distSq = p1.distanceSq(p2);
                    if (distSq < thresholdSq) {
                        double dist = Math.sqrt(distSq);
                        int alpha = (int) ((1.0 - (dist / CONNECTION_THRESHOLD)) * 80);
                        g2.setColor(WHITE_ALPHA[Math.min(255, Math.max(0, alpha))]);
                        g2.drawLine((int) p1.x, (int) p1.y, (int) p2.x, (int) p2.y);
                    }
                }
            }

            for (Particle p : particles) {
                int alphaInt = (int) (p.alpha * 255);
                g2.setColor(WHITE_ALPHA[Math.min(255, Math.max(0, alphaInt))]);
                int size = (int) p.size;
                g2.fillOval((int) (p.x - size / 2.0), (int) (p.y - size / 2.0), size, size);
            }
        }

        class Particle {
            double x, y;
            double vx, vy;
            double size;
            double alpha;

            Particle() {
                reset(true);
            }

            void reset(boolean randomizePos) {
                if (randomizePos) {
                    x = Math.random() * getStyleableWidth();
                    y = Math.random() * getStyleableHeight();
                } else {
                    x = Math.random() * getStyleableWidth();
                    y = Math.random() * getStyleableHeight();
                }
                double speed = 0.35 + Math.random() * 0.55;
                double angle = Math.random() * 2 * Math.PI;
                vx = Math.cos(angle) * speed;
                vy = Math.sin(angle) * speed;

                size = 2.0 + Math.random() * 2.5;
                alpha = 0.2 + Math.random() * 0.4;
            }

            double distanceSq(Particle other) {
                double dx = x - other.x;
                double dy = y - other.y;
                return dx * dx + dy * dy;
            }

            void update(int w, int h) {
                x += vx;
                y += vy;

                // Bounce off edges with a 10% buffer zone for smoother entry/exit
                double bufferX = w * 0.10;
                double bufferY = h * 0.10;

                if (x < -bufferX) {
                    x = -bufferX;
                    vx *= -1;
                } else if (x > w + bufferX) {
                    x = w + bufferX;
                    vx *= -1;
                }

                if (y < -bufferY) {
                    y = -bufferY;
                    vy *= -1;
                } else if (y > h + bufferY) {
                    y = h + bufferY;
                    vy *= -1;
                }
            }
        }

        @Override
        public void updateUI() {
            super.updateUI();
            updateHeaderForegrounds();
        }
    }

    // Toggle buttons
    private JToggleButton particleToggle;

    private JPanel createHeader() {
        // Use the new inner class
        DynamicHeaderPanel dhPanel = new DynamicHeaderPanel();
        headerPanel = dhPanel; // Assign to the field (which is JPanel type)

        headerPanel.setOpaque(false);
        headerPanel.setBorder(BorderFactory.createEmptyBorder(20, 25, 20, 25));

        // 4. Layout: Left Side (Logo & Title)
        headerTitlePanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 0));
        headerTitlePanel.setOpaque(false);

        // TODO: Add icon
        // Icon
        headerIconLabel = new JLabel("");
        try {
            java.net.URL iconUrl = getClass().getResource("/carafe-icon.png");
            if (iconUrl != null) {
                ImageIcon originalIcon = new ImageIcon(iconUrl);
                // Scale to a high-but-reasonable resolution (128x128) to support HiDPI up to
                // ~250%
                // without keeping the massive original in memory for every paint
                Image highResImage = originalIcon.getImage().getScaledInstance(128, 128, Image.SCALE_SMOOTH);

                headerIconLabel.setIcon(new javax.swing.Icon() {
                    @Override
                    public void paintIcon(java.awt.Component c, java.awt.Graphics g, int x, int y) {
                        java.awt.Graphics2D g2 = (java.awt.Graphics2D) g.create();
                        g2.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                                java.awt.RenderingHints.VALUE_INTERPOLATION_BICUBIC);
                        g2.setRenderingHint(java.awt.RenderingHints.KEY_RENDERING,
                                java.awt.RenderingHints.VALUE_RENDER_QUALITY);
                        g2.setRenderingHint(java.awt.RenderingHints.KEY_ANTIALIASING,
                                java.awt.RenderingHints.VALUE_ANTIALIAS_ON);

                        // Clip to rounded rectangle
                        java.awt.geom.RoundRectangle2D rounded = new java.awt.geom.RoundRectangle2D.Float(
                                x, y, 48, 48, 16, 16);
                        g2.setClip(rounded);

                        // Draw the 128x128 image into the 48x48 layout space
                        g2.drawImage(highResImage, x, y, 48, 48, null);
                        g2.dispose();
                    }

                    @Override
                    public int getIconWidth() {
                        return 48;
                    }

                    @Override
                    public int getIconHeight() {
                        return 48;
                    }
                });
            }
        } catch (Exception e) {
            // failed to load icon
        }
        headerIconLabel.setFont(new Font("Segoe UI", Font.BOLD, 42));

        headerTextPanel = new JPanel();
        headerTextPanel.setOpaque(false);
        headerTextPanel.setLayout(new BoxLayout(headerTextPanel, BoxLayout.Y_AXIS));

        headerTitleLabel = new JLabel("Carafe");
        headerTitleLabel.setFont(new Font("Segoe UI", Font.BOLD, 28));

        headerSubtitleLabel = new JLabel("AI-Powered Spectral Library Generator for DIA Proteomics");
        headerSubtitleLabel.setFont(new Font("Segoe UI", Font.PLAIN, 13));

        headerTextPanel.add(headerTitleLabel);
        headerTextPanel.add(Box.createVerticalStrut(3));
        headerTextPanel.add(headerSubtitleLabel);

        headerTitlePanel.add(headerIconLabel);
        headerTitlePanel.add(headerTextPanel);

        // 5. Layout: Right Side (Toggle & Version)
        headerRightPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 15, 0));
        headerRightPanel.setOpaque(false);

        // Reuse the generic creation method
        darkModeToggle = createHeaderToggleButton("Light Mode", false,
                e -> toggleDarkMode(darkModeToggle.isSelected()));

        // New Particle Toggle
        particleToggle = createHeaderToggleButton("Effects On", true, e -> {
            boolean selected = particleToggle.isSelected();
            particleToggle.setText(selected ? "Effects On" : "Effects Off");
            dhPanel.setAnimationEnabled(selected);
        });

        headerRightPanel.add(particleToggle);
        headerRightPanel.add(darkModeToggle);

        headerVersionLabel = new JLabel(CParameter.getVersion());
        headerVersionLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        headerRightPanel.add(headerVersionLabel);

        headerPanel.add(headerTitlePanel, BorderLayout.WEST);
        headerPanel.add(headerRightPanel, BorderLayout.EAST);

        // Sync initial state
        updateHeaderForegrounds();

        // Ensure buttons have correct label color initially
        updateHeaderForegrounds();

        return headerPanel;
    }

    /**
     * Updates the foreground colors of all header components based on
     * the current theme and background luminance.
     */
    private void updateHeaderForegrounds() {
        boolean dark = FlatLaf.isLafDark();
        Color base = lafColor("Carafe.headerBase", new Color(0x2F82B7));

        // Use mid-gradient color as reference for text contrast
        Color bgSample = dark ? adjust(base, -20) : adjust(base, 35);
        Color fgPrimary = pickOnColor(bgSample);

        if (headerIconLabel != null)
            headerIconLabel.setForeground(fgPrimary);
        if (headerTitleLabel != null)
            headerTitleLabel.setForeground(fgPrimary);
        if (headerSubtitleLabel != null)
            headerSubtitleLabel.setForeground(withAlpha(fgPrimary, 210));
        if (headerVersionLabel != null)
            headerVersionLabel.setForeground(withAlpha(fgPrimary, 180));

        if (darkModeToggle != null) {
            darkModeToggle.setSelected(dark);
            darkModeToggle.setText(dark ? "Light Mode" : "Dark Mode");
            // This ensures the font is never White-on-White if pickOnColor returns dark for
            // light backgrounds
            darkModeToggle.setForeground(fgPrimary);
        }

        if (particleToggle != null) {
            particleToggle.setText(particleToggle.isSelected() ? "Effects On" : "Effects Off");
            particleToggle.setForeground(fgPrimary);
        }

        if (headerPanel != null)
            headerPanel.repaint();
    }

    // ---------------- helpers ----------------

    private static Color adjust(Color c, int delta) {
        int r = Math.max(0, Math.min(255, c.getRed() + delta));
        int g = Math.max(0, Math.min(255, c.getGreen() + delta));
        int b = Math.max(0, Math.min(255, c.getBlue() + delta));
        return new Color(r, g, b);
    }

    private static Color withAlpha(Color c, int a) {
        return new Color(c.getRed(), c.getGreen(), c.getBlue(), Math.max(0, Math.min(255, a)));
    }

    private static Color pickOnColor(Color bg) {
        // relative luminance (sRGB)
        double r = bg.getRed() / 255.0;
        double g = bg.getGreen() / 255.0;
        double b = bg.getBlue() / 255.0;

        r = (r <= 0.03928) ? (r / 12.92) : Math.pow((r + 0.055) / 1.055, 2.4);
        g = (g <= 0.03928) ? (g / 12.92) : Math.pow((g + 0.055) / 1.055, 2.4);
        b = (b <= 0.03928) ? (b / 12.92) : Math.pow((b + 0.055) / 1.055, 2.4);

        double L = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        return (L > 0.70) ? new Color(20, 20, 20) : Color.WHITE;
    }

    private void toggleDarkMode(boolean isDark) {
        try {
            // 1. Set the new Look and Feel state globally
            if (isDark) {
                FlatDarkLaf.setup();
            } else {
                FlatLightLaf.setup();
            }
            customizeUIDefaults();

            // Persist preference FIRST so components reading it see the new state
            prefs.putBoolean(PREF_DARK_MODE, isDark);

            // 2. Fast refresh of the window contents (resets standard properties)
            SwingUtilities.updateComponentTreeUI(this);

            // 3. Update custom components (re-apply overrides)
            updateHeaderForegrounds();
            applyThemeToCustomComponents();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void updateConsoleTheme() {
        Color bg = UIManager.getColor("TextArea.background");
        if (bg == null)
            bg = UIManager.getColor("TextComponent.background");

        Color fg = UIManager.getColor("TextArea.foreground");
        if (fg == null)
            fg = UIManager.getColor("TextComponent.foreground");

        Color caret = UIManager.getColor("TextArea.caretForeground");
        if (caret == null)
            caret = fg;

        consoleArea.setOpaque(true);
        if (bg != null)
            consoleArea.setBackground(bg);
        if (fg != null)
            consoleArea.setForeground(fg);
        consoleArea.setCaretColor(caret);

        if (consoleScrollPane.getViewport() != null) {
            consoleScrollPane.getViewport().setOpaque(true);
            if (bg != null)
                consoleScrollPane.getViewport().setBackground(bg);
        }

        Color spBg = UIManager.getColor("ScrollPane.background");
        if (spBg != null)
            consoleScrollPane.setBackground(spBg);
    }

    /**
     * Optimized sync method.
     */
    private void applyThemeToCustomComponents() {
        if (infoCards != null) {
            for (InfoCardRef ref : infoCards) {
                updateInfoCardTheme(ref);
            }
        }

        // Refresh styles for file input fields (hyperlinks)
        if (trainMsFileField != null && trainMsFiles != null) {
            updateFileFieldState(trainMsFileField, trainMsFiles);
        }
        if (projectMsFileField != null && projectMsFiles != null) {
            updateFileFieldState(projectMsFileField, projectMsFiles);
        }

        refreshStatusLabel();
    }

    private void restyleButtonsRecursively(java.awt.Container root) {
        if (root == null)
            return;

        for (java.awt.Component c : root.getComponents()) {
            if (c instanceof JButton b) {
                Object role = b.getClientProperty("carafe.role");
                if ("generic".equals(role)) {
                    styleButton(b);
                } else if ("secondary".equals(role)) {
                    styleSecondaryButton(b);
                } else if ("primary".equals(role)) {
                    // Keep primary buttons as they are (brand color)
                    // but ensure text remains readable
                    b.setForeground(Color.WHITE);
                    b.setOpaque(true);
                }
            } else if (c instanceof java.awt.Container child) {
                restyleButtonsRecursively(child);
            }
        }
    }

    private void updateInfoCardTheme(InfoCardRef ref) {
        boolean dark = FlatLaf.isLafDark();

        Color bg = dark ? new Color(0x2B333A) : new Color(0xE7F3FF);
        Color border = dark ? new Color(0x55616B) : new Color(0x8BBBE6);
        Color titleFg = dark ? new Color(0x7CC7FF) : new Color(0x2A78B8);
        Color textFg = dark ? new Color(0xD6DEE6) : UIManager.getColor("Label.foreground");

        // card
        ref.card.setOpaque(true);
        ref.card.setBackground(bg);

        ref.card.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(border, 1, false),
                BorderFactory.createEmptyBorder(16, 16, 16, 16)));

        // title
        ref.titleLabel.setForeground(titleFg);

        // text area
        ref.contentArea.setOpaque(false);
        ref.contentArea.setForeground(textFg);
        ref.contentArea.setCaretColor(textFg);
    }

    private JPanel createInputPanel() {
        JPanel panel = new ScrollablePanel(new BorderLayout());
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        // Top section: Workflow selection
        JPanel workflowPanel = new JPanel(new GridBagLayout());
        GridBagConstraints wgbc = new GridBagConstraints();
        wgbc.fill = GridBagConstraints.HORIZONTAL;
        wgbc.insets = new Insets(0, COL_SPACING, 15, COL_SPACING);
        wgbc.anchor = GridBagConstraints.EAST;

        wgbc.gridx = 0;
        wgbc.gridy = 0;
        wgbc.weightx = 0;
        workflowPanel.add(createLabel("Workflow:"), wgbc);

        String[] workflows = {
                "1. Spectral library generation: start with DIA-NN search",
                "2. Spectral library generation: start with DIA-NN report",
                "3. End-to-end DIA search"
        };
        workflowCombo = new JComboBox<>(workflows);
        styleComboBox(workflowCombo);
        workflowCombo.setToolTipText("Select your workflow type");
        workflowCombo.addActionListener(e -> updateInputFieldsVisibility());
        wgbc.gridx = 1;
        wgbc.weightx = 1;
        wgbc.gridwidth = 2;
        workflowPanel.add(workflowCombo, wgbc);

        panel.add(workflowPanel, BorderLayout.NORTH);

        // Center section: Dynamic input fields
        inputFieldsPanel = new JPanel(new GridBagLayout());

        int gridy = 0;

        trainMsRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Train MS File:",
                "MS/MS data for model training.\n" +
                        "Supported formats: mzML, Thermo raw, Bruker raw (.d).\n" +
                        "A single MS/MS file, multiple MS/MS files, or a folder containing MS/MS files are accepted.\n"
                        +
                        "Thermo raw files are only supported when starting with DIA-NN search or performing end-to-end DIA search.\n"
                        +
                        "When the format is Thermo raw, MSConvert (ProteoWizard) needs to be installed (convert raw to mzML for Carafe).",
                trainMsFileField = createTextField("Path to mzML/raw file or folder for training"),
                createMsButtonsPanel(trainMsFileField));

        // Initialize debounce timer (300ms delay)
        if (msConvertVisibilityDebounceTimer == null) {
            msConvertVisibilityDebounceTimer = new javax.swing.Timer(300, e -> updateMsConvertVisibility());
            msConvertVisibilityDebounceTimer.setRepeats(false);
        }

        trainMsFileField.getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
            @Override
            public void insertUpdate(javax.swing.event.DocumentEvent e) {
                msConvertVisibilityDebounceTimer.restart();
            }

            @Override
            public void removeUpdate(javax.swing.event.DocumentEvent e) {
                msConvertVisibilityDebounceTimer.restart();
            }

            @Override
            public void changedUpdate(javax.swing.event.DocumentEvent e) {
                msConvertVisibilityDebounceTimer.restart();
            }
        });

        diannReportRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "DIA-NN Report:",
                "A peptide detection file used for model training.\n" +
                        "The main report file from DIA-NN (from v1.8.1 to v2.x.x) is supported.\n" +
                        "Supported formats: tsv, parquet. (e.g. report.tsv or report.parquet)\n" +
                        "This file must be directly generated using the same input train MS file(s).",
                diannReportFileField = createTextField("Path to DIA-NN report.tsv or report.parquet"),
                createBrowseButton(diannReportFileField, "DIA-NN Report", new String[] { "tsv", "parquet" }));

        trainDbRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Train Protein Database:",
                "Protein database used for peptide detection on the train MS file(s).\n" +
                        "Supported formats: FASTA. (e.g. protein.fasta or protein.fa)",
                trainDbFileField = createTextField("Path to protein FASTA for training"),
                createBrowseButton(trainDbFileField, "FASTA Files", new String[] { "fasta", "fa" }));

        // This is the MS/MS data for peptide detection using the fine-tuned spectral
        // library
        projectMsRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Project MS File:",
                "MS/MS data for peptide detection using the fine-tuned spectral library using DIA-NN.\n" +
                        "Supported formats: mzML, Thermo raw, Bruker raw (.d).\n" +
                        "A single MS/MS file, multiple MS/MS files, or a folder containing MS/MS files are accepted.\n"
                        +
                        "When the format is Thermo raw, users need to make sure DIA-NN is configured to use Thermo raw format.",
                projectMsFileField = createTextField("Path to mzML/raw file or folder for project"),
                createMsButtonsPanel(projectMsFileField));

        libraryDbRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Library Protein Database:",
                "Protein database used for fine-tuned spectral library generation.\n" +
                        "Supported formats: FASTA. (e.g. protein.fasta or protein.fa)",
                libraryDbFileField = createTextField("Path to protein FASTA for library generation"),
                createBrowseButton(libraryDbFileField, "FASTA Files", new String[] { "fasta", "fa" }));

        addInputRowToPanel(inputFieldsPanel, gridy++, "Output Directory:",
                "Output directory for the analysis.",
                outputDirField = createTextField("Path to output directory"),
                createFolderButton(outputDirField));

        addInputRowToPanel(inputFieldsPanel, gridy++, "Python Executable:",
                "Python path (the path of python.exe (Windows) or python (Linux/Mac)) for Carafe model fine-tuning.\n" +
                        "Carafe requires a customized AlphaPeptDeep python package for model fine-tuning.\n" +
                        "Users can install all the dependent python packages by clicking the 'Install' button.",
                pythonPathCombo = createPythonComboBox(),
                createPythonBrowseButton());

        diannExeRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "DIA-NN Executable:",
                "DIA-NN path (the path of diann.exe (NOT DIA-NN.exe) (Windows) or diann (Linux/Mac)).",
                diannPathCombo = createDiannComboBox(),
                createDiannBrowseButton());

        msConvertExeRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "MSConvert Executable:",
                "MSConvert path (the path of msconvert.exe (NOT MSConvertGUI.exe) (Windows).",
                msConvertPathCombo = createMsConvertComboBox(),
                createMsConvertBrowseButton());

        diannAdditionalOptionsRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++,
                "DIA-NN additional options:",
                "Additional command line options for DIA-NN.",
                diannAdditionalOptionsField = createTextField("DIA-NN additional options"),
                null);

        addInputRowToPanel(inputFieldsPanel, gridy++, "Carafe additional options:",
                "Additional command line options for Carafe.",
                carafeAdditionalOptionsField = createTextField("Carafe additional options"),
                null);

        JPanel infoWrapper = new JPanel(new BorderLayout());
        infoWrapper.add(createInfoCard(
                "Workflow Guide",
                "Workflow 1: Generate spectral library by running DIA-NN search first\n" +
                        "  - Requires: Train MS files, Train database, Library database\n\n" +
                        "Workflow 2: Generate spectral library from existing DIA-NN results\n" +
                        "  - Requires: DIA-NN report file, Train MS files, Library database\n\n" +
                        "Workflow 3: Complete DIA analysis pipeline (Carafe+DIA-NN)\n" +
                        "  - Requires: Train MS, Project MS, both databases"),
                BorderLayout.CENTER);

        GridBagConstraints gbcInfo = new GridBagConstraints();
        gbcInfo.gridx = 0;
        gbcInfo.gridy = gridy++;
        gbcInfo.gridwidth = 3;
        gbcInfo.fill = GridBagConstraints.HORIZONTAL;
        gbcInfo.insets = new Insets(15, COL_SPACING, 0, COL_SPACING);
        inputFieldsPanel.add(infoWrapper, gbcInfo);

        GridBagConstraints gbcGlue = new GridBagConstraints();
        gbcGlue.gridy = gridy;
        gbcGlue.weighty = 1.0;
        inputFieldsPanel.add(Box.createVerticalGlue(), gbcGlue);

        panel.add(inputFieldsPanel, BorderLayout.CENTER);

        updateInputFieldsVisibility();
        return panel;
    }

    private java.util.List<JComponent> addInputRowToPanel(JPanel container, int gridy, String labelText,
            String toolTipText,
            JComponent inputField, JComponent buttonComponent) {
        java.util.List<JComponent> rowComponents = new java.util.ArrayList<>();
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridy = gridy;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(ROW_SPACING, COL_SPACING, ROW_SPACING, COL_SPACING);
        gbc.anchor = GridBagConstraints.EAST;

        gbc.gridx = 0;
        gbc.weightx = 0;
        JLabel label = createLabel(labelText, toolTipText);
        container.add(label, gbc);
        rowComponents.add(label);

        gbc.gridx = 1;
        gbc.weightx = 1;
        if (buttonComponent == null) {
            gbc.gridwidth = 2;
        }
        container.add(inputField, gbc);
        rowComponents.add(inputField);
        gbc.gridwidth = 1;

        if (buttonComponent != null) {
            gbc.gridx = 2;
            gbc.weightx = 0;
            container.add(buttonComponent, gbc);
            rowComponents.add(buttonComponent);
        }

        return rowComponents;
    }

    private JPanel createMsButtonsPanel(JTextField targetField) {
        JPanel msButtonsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 2, 0));

        JButton browse;
        final java.util.List<String> associatedList;

        if (targetField == trainMsFileField) {
            associatedList = trainMsFiles;
            browse = createMultiFileBrowseButton(targetField, "mzML/raw Files", new String[] { "mzML", "raw" },
                    associatedList);
            setupMultiFileFieldInteraction(targetField, associatedList);
        } else if (targetField == projectMsFileField) {
            associatedList = projectMsFiles;
            browse = createMultiFileBrowseButton(targetField, "mzML/raw Files", new String[] { "mzML", "raw" },
                    associatedList);
            setupMultiFileFieldInteraction(targetField, associatedList);
        } else {
            associatedList = null;
            browse = createBrowseButton(targetField, "mzML/raw Files", new String[] { "mzML", "raw" });
        }

        msButtonsPanel.add(browse);

        // Custom Folder Button logic to ensure list is cleared
        JButton folderBtn = new JButton("Folder");
        styleButton(folderBtn);
        folderBtn.addActionListener(e -> {
            setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
            new Thread(() -> {
                try {
                    JFileChooser chooser = new JFileChooser();
                    String lastDir = prefs.get(PREF_LAST_DIR, System.getProperty("user.home"));
                    chooser.setCurrentDirectory(new File(lastDir));
                    chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                    SwingUtilities.invokeLater(() -> {
                        setCursor(Cursor.getDefaultCursor());
                        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                            File selectedDir = chooser.getSelectedFile();
                            if (associatedList != null) {
                                associatedList.clear();
                                updateFileFieldState(targetField, associatedList);
                                // updateFileFieldState with empty list sets text to empty usually but we want
                                // to set it to folder path.
                                // So we must manually set text.
                                targetField.setText(selectedDir.getAbsolutePath());
                                // Force state to "Single File/Folder" mode manually since updateFileFieldState
                                // sees empty list
                                targetField.setForeground(UIManager.getColor("TextField.foreground"));
                                targetField.setCursor(Cursor.getPredefinedCursor(Cursor.TEXT_CURSOR));
                                targetField.setEditable(true);
                            } else {
                                targetField.setText(selectedDir.getAbsolutePath());
                            }
                            prefs.put(PREF_LAST_DIR, selectedDir.getAbsolutePath());
                        }
                    });
                } catch (Exception ex) {
                    SwingUtilities.invokeLater(() -> setCursor(Cursor.getDefaultCursor()));
                    ex.printStackTrace();
                }
            }).start();
        });

        msButtonsPanel.add(folderBtn);
        return msButtonsPanel;
    }

    private JButton createMultiFileBrowseButton(JTextField targetField, String description, String[] extensions,
            java.util.List<String> fileList) {
        JButton button = new JButton("Browse");
        styleButton(button);
        button.addActionListener(e -> {
            chooseFiles("Select Files", extensions, description, files -> {
                // Validation: Check for mixed extensions
                boolean hasMzML = false;
                boolean hasRaw = false;
                for (File f : files) {
                    String name = f.getName().toLowerCase();
                    if (name.endsWith(".mzml"))
                        hasMzML = true;
                    if (name.endsWith(".raw"))
                        hasRaw = true;
                }
                if (hasMzML && hasRaw) {
                    JOptionPane.showMessageDialog(this,
                            "Please select only mzML files OR only RAW files, not both.",
                            "Invalid Selection",
                            JOptionPane.WARNING_MESSAGE);
                    return;
                }

                fileList.clear();
                for (File f : files) {
                    fileList.add(f.getAbsolutePath());
                }
                updateFileFieldState(targetField, fileList);
                prefs.put(PREF_LAST_DIR, files[0].getParent());
            });
        });
        return button;
    }

    private void updateFileFieldState(JTextField field, java.util.List<String> files) {
        java.util.Map<java.awt.font.TextAttribute, Object> attributes = new java.util.HashMap<>(
                field.getFont().getAttributes());
        if (files != null && files.size() > 1) {
            field.setEditable(false);
            field.setText("(" + files.size() + " files selected)");

            // Theme-aware hyperlink color
            boolean isDark = prefs.getBoolean(PREF_DARK_MODE, false);
            if (isDark) {
                field.setForeground(new Color(100, 180, 255)); // Light Blue for Dark Mode
            } else {
                field.setForeground(Color.BLUE);
            }

            field.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            attributes.put(java.awt.font.TextAttribute.UNDERLINE, java.awt.font.TextAttribute.UNDERLINE_ON);
        } else {
            field.setForeground(UIManager.getColor("TextField.foreground"));
            field.setCursor(Cursor.getPredefinedCursor(Cursor.TEXT_CURSOR));
            field.setEditable(true);
            attributes.put(java.awt.font.TextAttribute.UNDERLINE, -1);
            if (files != null && files.size() == 1) {
                field.setText(files.get(0));
            }
            // Do NOT clear text if list is empty, to preserve manual input
        }
        field.setFont(field.getFont().deriveFont(attributes));
    }

    private void setupMultiFileFieldInteraction(JTextField field, java.util.List<String> fileList) {
        // Remove existing listeners if any? No easy way, assuming called once.

        field.getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
            private void update() {
                if (field.isEditable()) {
                    // If user is editing text manually, and it doesn't match the known single file,
                    // or if list has multiple files (which shouldn't happen if editable, but safety
                    // check),
                    // we clear the list to rely on text field content.
                    if (!fileList.isEmpty()) {
                        String text = field.getText();
                        if (fileList.size() > 1) {
                            // Should not be editable if multiple files!
                            // But if it happened somehow, clear list.
                            fileList.clear();
                        } else if (fileList.size() == 1) {
                            if (!text.equals(fileList.get(0))) {
                                fileList.clear();
                            }
                        }
                    }
                }
            }

            @Override
            public void insertUpdate(javax.swing.event.DocumentEvent e) {
                update();
            }

            @Override
            public void removeUpdate(javax.swing.event.DocumentEvent e) {
                update();
            }

            @Override
            public void changedUpdate(javax.swing.event.DocumentEvent e) {
                update();
            }
        });

        field.addMouseListener(new java.awt.event.MouseAdapter() {
            @Override
            public void mouseClicked(java.awt.event.MouseEvent e) {
                if (!field.isEditable()) {
                    // Summary Mode: Single Click opens dialog
                    if (e.getClickCount() == 1) {
                        showFileListDialog(field, fileList);
                    }
                }
            }
        });
    }

    private void showFileListDialog(JTextField field, java.util.List<String> fileList) {
        String title = isRunning ? "View File List" : "Edit File List";
        javax.swing.JDialog d = new javax.swing.JDialog(this, title, true);
        d.setSize(600, 400);
        d.setLocationRelativeTo(this);
        d.setLayout(new BorderLayout());

        JTextArea textArea = new JTextArea();
        // If running, make read-only
        if (isRunning) {
            textArea.setEditable(false);
        }

        if (fileList.isEmpty() && !field.getText().trim().isEmpty() && !field.getText().startsWith("(")) {
            // Populate with single/folder path from text field if list is empty
            textArea.setText(field.getText().trim());
        } else {
            for (String path : fileList) {
                textArea.append(path + "\n");
            }
        }

        d.add(new JScrollPane(textArea), BorderLayout.CENTER);

        JPanel btnPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton okBtn = new JButton("OK");
        okBtn.setEnabled(!isRunning); // Disable OK button if running

        okBtn.addActionListener(ev -> {
            if (isRunning) {
                // Double safety
                d.dispose();
                return;
            }

            // Validate lines
            String[] lines = textArea.getText().split("\\n");
            java.util.List<String> newPaths = new java.util.ArrayList<>();
            boolean hasMzML = false;
            boolean hasRaw = false;

            for (String line : lines) {
                String path = line.trim();
                if (!path.isEmpty()) {
                    newPaths.add(path);
                    if (path.toLowerCase().endsWith(".mzml"))
                        hasMzML = true;
                    if (path.toLowerCase().endsWith(".raw"))
                        hasRaw = true;
                }
            }

            if (hasMzML && hasRaw) {
                JOptionPane.showMessageDialog(d,
                        "Please select only mzML files OR only RAW files, not both.",
                        "Invalid Selection",
                        JOptionPane.WARNING_MESSAGE);
                return;
            }

            fileList.clear();
            fileList.addAll(newPaths);
            updateFileFieldState(field, fileList);
            d.dispose();
        });

        JButton cancelBtn = new JButton(isRunning ? "Close" : "Cancel");
        cancelBtn.addActionListener(ev -> d.dispose());

        btnPanel.add(okBtn);
        btnPanel.add(cancelBtn);
        d.add(btnPanel, BorderLayout.SOUTH);
        d.setVisible(true); // Modal, so execution blocks here
    }

    private void setVisible(java.util.List<JComponent> components, boolean visible) {
        if (components != null) {
            for (JComponent component : components) {
                component.setVisible(visible);
            }
        }
    }

    private void updateInputFieldsVisibility() {
        globalWorkflowIndex = workflowCombo.getSelectedIndex();

        switch (globalWorkflowIndex) {
            case 0 -> { // Workflow 1
                setVisible(diannReportRowComponents, false);
                setVisible(trainMsRowComponents, true);
                setVisible(trainDbRowComponents, true);
                setVisible(projectMsRowComponents, false);
                setVisible(libraryDbRowComponents, true);
                setVisible(diannExeRowComponents, true);
                setVisible(diannAdditionalOptionsRowComponents, true);
                updateMsConvertVisibility();
            }
            case 1 -> { // Workflow 2
                setVisible(diannReportRowComponents, true);
                setVisible(trainMsRowComponents, true);
                setVisible(trainDbRowComponents, false);
                setVisible(projectMsRowComponents, false);
                setVisible(libraryDbRowComponents, true);
                setVisible(diannExeRowComponents, false);
                setVisible(diannAdditionalOptionsRowComponents, false);
                updateMsConvertVisibility();
            }
            case 2 -> { // Workflow 3
                setVisible(diannReportRowComponents, false);
                setVisible(trainMsRowComponents, true);
                setVisible(trainDbRowComponents, true);
                setVisible(projectMsRowComponents, true);
                setVisible(libraryDbRowComponents, true);
                setVisible(diannExeRowComponents, true);
                setVisible(diannAdditionalOptionsRowComponents, true);
                updateMsConvertVisibility();
            }
        }

        stripScrollPaneBorder(inputFieldsPanel);
        inputFieldsPanel.setBorder(BorderFactory.createEmptyBorder());
        inputFieldsPanel.revalidate();
        inputFieldsPanel.repaint();
    }

    private JPanel createTrainingDataPanel() {
        JPanel panel = new ScrollablePanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = DEFAULT_INSETS;
        gbc.anchor = GridBagConstraints.WEST;

        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.weightx = 0;
        panel.add(createLabel("False Discovery Rate:",
                "The false discovery rate threshold (or q-value) for peptide precursor filtering."), gbc);

        fdrSpinner = createDoubleSpinner(0.01, 0.001, 0.1, 0.005);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(fdrSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.weightx = 0;
        panel.add(
                createLabel("PTM Site Probability:",
                        "The site probability threshold for PTM peptideform detection filtering.\n" +
                                "This is used when fine-tuning models for PTM dataset such as phosphoproteomics data."),
                gbc);

        ptmSiteProbSpinner = createDoubleSpinner(0.75, 0.0, 1.0, 0.05);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(ptmSiteProbSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.weightx = 0;
        panel.add(createLabel("PTM Site Q-value:", "The q-value threshold for PTM peptideform detection filtering.\n" +
                "This is used when fine-tuning models for PTM dataset such as phosphoproteomics data."), gbc);

        ptmSiteQvalueSpinner = createDoubleSpinner(0.01, 0.001, 0.1, 0.005);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(ptmSiteQvalueSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 3;
        gbc.weightx = 0;
        panel.add(createLabel("Fragment Ion Mass Tolerance:",
                "The mass tolerance for fragment ion mass tolerance used\n" +
                        "during fragment ion intensity annotation and XIC extraction."),
                gbc);

        fragTolSpinner = createSpinner(20, 1, 100, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(fragTolSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 4;
        gbc.weightx = 0;
        panel.add(createLabel("Fragment Ion Mass Tolerance Units:",
                "The mass tolerance unit for fragment ion mass tolerance."), gbc);

        String[] tolUnits = { "ppm", "Da" };
        fragTolUnitCombo = new JComboBox<>(tolUnits);
        styleComboBox(fragTolUnitCombo);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(fragTolUnitCombo, gbc);

        gbc.gridx = 0;
        gbc.gridy = 5;
        gbc.weightx = 0;
        panel.add(createLabel("Refine Peak Boundaries:",
                "Refine the peak boundaries for peptide-centric shared fragment ion detection.\n" +
                        "If uncheck, the peak boundaries will be set based on the input peptide detection file."),
                gbc);

        refineBoundaryCheckbox = createCheckBox("", true);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(refineBoundaryCheckbox, gbc);

        gbc.gridx = 0;
        gbc.gridy = 6;
        gbc.weightx = 0;
        panel.add(createLabel("Peak refinement RT Window:", "RT window for refine peak boundary in minute.\n" +
                "This is used to refine the peak boundaries for\n" +
                "peptide-centric shared fragment ion detection.\n" +
                "Set to 'auto' to set it based on LC gradient length."), gbc);

        rtPeakWindowField = createTextField("auto");
        rtPeakWindowField.setText("auto");
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(rtPeakWindowField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 7;
        gbc.weightx = 0;
        panel.add(createLabel("XIC Correlation:",
                "The correlation threshold for fragment ion to be considered as valid\n" +
                        "for fragment ion intensity model finetuning."),
                gbc);

        xicCorSpinner = createDoubleSpinner(0.8, 0.0, 1.0, 0.05);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(xicCorSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 8;
        gbc.weightx = 0;
        panel.add(createLabel("Minimum Fragment m/z:", "The minimum fragment ion m/z to consider to be valid"), gbc);

        // min_fragment_ion_mz
        minFragMzSpinner = createSpinner(200, 50, 500, 10);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(minFragMzSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 9;
        gbc.weightx = 0;
        panel.add(createLabel("N-term min ion:",
                "For N-terminal fragment ions (such as b-ion) with number <= n_ion_min, they will be considered as invalid."),
                gbc);

        nIonMinSpinner = createSpinner(2, 0, 3, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(nIonMinSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 10;
        gbc.weightx = 0;
        panel.add(createLabel("C-term min ion:",
                "For C-terminal fragment ions (such as y-ion) with number <= c_ion_min, they will be considered as invalid."),
                gbc);

        cIonMinSpinner = createSpinner(2, 0, 3, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(cIonMinSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 11;
        gbc.gridwidth = 2;
        gbc.weighty = 1;
        panel.add(Box.createVerticalGlue(), gbc);

        return panel;
    }

    private JPanel createModelTrainingPanel() {
        JPanel panel = new ScrollablePanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = DEFAULT_INSETS;
        gbc.anchor = GridBagConstraints.WEST;

        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.weightx = 0;
        panel.add(createLabel("Model Type:", "The model type to use for model finetuning.\n" +
                "For global proteome, use the 'general' model.\n" +
                "For phosphoproteome, use the 'phosphorylation' model."), gbc);

        String[] modes = { "general", "phosphorylation" };
        modeCombo = new JComboBox<>(modes);
        styleComboBox(modeCombo);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(modeCombo, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.weightx = 0;
        panel.add(createLabel("Normalized Collision Energy:",
                "The normalized collision energy (NCE) to use for deep learning model training and inference.\n" +
                        "NCE is one of the inputs to the deep learning model\n" +
                        "for fragment ion intensity model training and inference.\n" +
                        "When it is set to 'auto', Carafe will determine the NCE from the training MS/MS data and use it."),
                gbc);
        nceField = createTextField("e.g., 27 or auto");
        nceField.setText("auto");
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(nceField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.weightx = 0;
        panel.add(createLabel("MS Instrument Type:",
                "The MS instrument type to use for deep learning model training and inference.\n" +
                        "MS instrument type is one of the inputs to the deep learning model\n" +
                        "for fragment ion intensity model training and inference.\n" +
                        "When it is set to 'auto', Carafe will determine the instrument type from the training MS/MS data and use it."),
                gbc);

        String[] msInstruments = { "auto", "QE", "Lumos", "timsTOF", "SciexTOF", "ThermoTOF" };
        msInstrumentField = new JComboBox<>(msInstruments);
        msInstrumentField.setEditable(false);
        styleComboBox(msInstrumentField);
        msInstrumentField.setSelectedItem("auto");
        msInstrumentField.setToolTipText("Select MS instrument (one of auto, QE, Lumos, timsTOF, SciexTOF, ThermoTOF)");
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(msInstrumentField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 3;
        gbc.weightx = 0;
        panel.add(createLabel("Computational Device:",
                "The computational device to use for deep learning model training and inference.\n" +
                        "GPU is recommended for faster training (requires CUDA-compatible GPU)\n" +
                        "If GPU is not available, Carafe will automatically fall back to CPU.\n" +
                        "When it is set to 'auto', Carafe will automatically detect the available device and use it."),
                gbc);

        String[] devices = { "auto", "gpu", "cpu" };
        deviceCombo = new JComboBox<>(devices);
        deviceCombo.setEditable(false);
        styleComboBox(deviceCombo);
        deviceCombo.setSelectedItem("auto");
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(deviceCombo, gbc);

        gbc.gridx = 0;
        gbc.gridy = 4;
        gbc.gridwidth = 2;
        gbc.weightx = 1;
        gbc.insets = new Insets(20, 8, 8, 8);
        panel.add(createInfoCard(
                "Model Training Tips",
                "- GPU mode is recommended for faster training (requires CUDA-compatible GPU)\n" +
                        "- If GPU is not available, the software will automatically fall back to CPU\n" +
                        "- NCE and MS Instrument are optional for fine-tuning (learned from data)\n" +
                        "- Use 'phosphorylation' mode for phosphopeptide analysis"),
                gbc);

        gbc.gridy = 5;
        gbc.weighty = 1;
        panel.add(Box.createVerticalGlue(), gbc);

        return panel;
    }

    private JPanel createLibraryGenerationPanel() {
        JPanel panel = new ScrollablePanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = DEFAULT_INSETS;
        gbc.anchor = GridBagConstraints.WEST;

        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.weightx = 0;
        panel.add(createLabel("Enzyme:",
                "The enzyme to consider for protein in-silico digestion during library generation."), gbc);

        String[] enzymes = {
                "1: Trypsin (default)",
                "2: Trypsin (no P rule)",
                "3: Arg-C",
                "4: Arg-C (no P rule)",
                "5: Arg-N",
                "6: Glu-C",
                "7: Lys-C",
                "0: Non enzyme"
        };
        enzymeCombo = new JComboBox<>(enzymes);
        styleComboBox(enzymeCombo);
        enzymeCombo.setSelectedIndex(1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(enzymeCombo, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.weightx = 0;
        panel.add(createLabel("Missed Cleavages:", "The maximum number of missed cleavages to consider\n" +
                " for protein in-silico digestion during library generation."), gbc);

        missCleavageSpinner = createSpinner(1, 0, 5, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(missCleavageSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.weightx = 0;
        panel.add(createLabel("Fixed Modification Available:",
                "The available fixed modifications to consider for library generation.\n" +
                        "Each modification is represented by an integer number.\n" +
                        "For example, 0 means no modification, 1 means Carbamidomethyl of C, etc."),
                gbc);

        // Populate Fixed Modifications dynamically
        LinkedHashMap<Integer, String> mod_id2name = CModification.getInstance().get_top_mod_list(26);
        Vector<String> fixModItems = new Vector<>();
        fixModItems.add("0 - no modification");
        for (Map.Entry<Integer, String> entry : mod_id2name.entrySet()) {
            fixModItems.add(entry.getKey() + " - " + entry.getValue());
        }
        fixModAvailableCombo = new JComboBox<>(fixModItems);
        styleComboBox(fixModAvailableCombo);

        // Auto-select based on default value "1" (Carbamidomethylation) if possible
        for (int i = 0; i < fixModAvailableCombo.getItemCount(); i++) {
            if (fixModAvailableCombo.getItemAt(i).startsWith("1 -")) {
                fixModAvailableCombo.setSelectedIndex(i);
                break;
            }
        }

        fixModAvailableCombo.addActionListener(e -> {
            String selected = (String) fixModAvailableCombo.getSelectedItem();
            if (selected != null) {
                String[] parts = selected.split(" - ");
                if (parts.length > 0) {
                    fixModSelectedField.setText(parts[0]);
                }
            }
        });

        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(fixModAvailableCombo, gbc);

        gbc.gridx = 0;
        gbc.gridy = 3;
        gbc.weightx = 0;
        panel.add(createLabel("Fixed Modifications Selected:",
                "The selected fixed modifications to consider for library generation.\n" +
                        "Each modification is represented by an integer number.\n" +
                        "For example, 0 means no modification, 1 means Carbamidomethyl of C, etc.\n" +
                        "Multiple modifications can be selected by separating them with commas (e.g., 1,11,12)."),
                gbc);

        fixModSelectedField = createTextField("e.g., 1");
        fixModSelectedField.setText("1");
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(fixModSelectedField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 4;
        gbc.weightx = 0;
        panel.add(createLabel("Variable Modifications Available:",
                "The available variable modifications to consider for library generation.\n"
                        + "Each modification is represented by an integer number.\n" +
                        "For example, 0 means no modification, 2 means Oxidation of M,\n" +
                        "7 means Phospho of S, \"7,8,9\" means Phospho of S, T and Y, etc."),
                gbc);

        // Populate Variable Modifications with presets + dynamic list
        Vector<String> varModItems = new Vector<>();
        varModItems.add("0 - no modification");
        varModItems.add("7,8,9 - Phosphorylation (STY)");
        varModItems.add("2,7,8,9 - Oxidation (M) + Phosphorylation (STY)");

        for (Map.Entry<Integer, String> entry : mod_id2name.entrySet()) {
            String item = entry.getKey() + " - " + entry.getValue();
            // Check if this item is already covered by presets to avoid redundancy if
            // desired?
            // User requested appending the list, so we append all.
            varModItems.add(item);
        }

        varModAvailableCombo = new JComboBox<>(varModItems);
        styleComboBox(varModAvailableCombo);

        varModAvailableCombo.addActionListener(e -> {
            String selected = (String) varModAvailableCombo.getSelectedItem();
            if (selected != null) {
                String[] parts = selected.split(" - ");
                if (parts.length > 0) {
                    varModSelectedField.setText(parts[0]);
                }
            }
        });

        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(varModAvailableCombo, gbc);

        gbc.gridx = 0;
        gbc.gridy = 5;
        gbc.weightx = 0;
        panel.add(createLabel("Variable Modifications Selected:",
                "The selected variable modifications to consider for library generation.\n" +
                        "Each modification is represented by an integer number.\n" +
                        "For example, 0 means no modification, 2 means Oxidation of M,\n" +
                        "7 means Phospho of S, \"7,8,9\" means Phospho of S, T and Y, etc.\n" +
                        "Multiple modifications can be selected by separating them with commas (e.g., 2,7,8,9)."),
                gbc);

        varModSelectedField = createTextField("e.g., 0 or 2");
        varModSelectedField.setText("0");
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(varModSelectedField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 6;
        gbc.weightx = 0;
        panel.add(createLabel("Maximum Variable Modifications:",
                "The maximum number of variable modifications to consider for library generation."), gbc);

        maxVarSpinner = createSpinner(1, 0, 5, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(maxVarSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 7;
        gbc.weightx = 0;
        panel.add(createLabel("Clip N-Terminal Methionine:", "When digesting a protein starting with amino acid M,\n" +
                "two copies of the leading peptides (with and without the N-terminal M) are considered if checked."),
                gbc);

        clipNmCheckbox = createCheckBox("", true);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(clipNmCheckbox, gbc);

        gbc.gridx = 0;
        gbc.gridy = 8;
        gbc.weightx = 0;
        panel.add(createLabel("Minimum Peptide Length:",
                "The minimum length of peptide to consider for library generation."), gbc);

        minLengthSpinner = createSpinner(7, 1, 50, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(minLengthSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 9;
        gbc.weightx = 0;
        panel.add(createLabel("Maximum Peptide Length:",
                "The maximum length of peptide to consider for library generation."), gbc);

        maxLengthSpinner = createSpinner(35, 1, 100, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(maxLengthSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 10;
        gbc.weightx = 0;
        panel.add(createLabel("Minimum Peptide m/z:", "The minimum m/z of peptide to consider for library generation.\n"
                +
                "This setting will be changed based on the minimum precursor m/z detected in the training MS/MS data."),
                gbc);

        minPepMzSpinner = createSpinner(300, 100, 2000, 50);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(minPepMzSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 11;
        gbc.weightx = 0;
        panel.add(createLabel("Maximum Peptide m/z:", "The maximum m/z of peptide to consider for library generation.\n"
                +
                "This setting will be changed based on the maximum precursor m/z detected in the training MS/MS data."),
                gbc);

        maxPepMzSpinner = createSpinner(1800, 100, 3000, 50);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(maxPepMzSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 12;
        gbc.weightx = 0;
        panel.add(createLabel("Minimum Peptide Charge:",
                "The minimum charge of peptide to consider for library generation."), gbc);

        minPepChargeSpinner = createSpinner(2, 1, 10, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(minPepChargeSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 13;
        gbc.weightx = 0;
        panel.add(createLabel("Maximum Peptide Charge:",
                "The maximum charge of peptide to consider for library generation."), gbc);

        maxPepChargeSpinner = createSpinner(4, 1, 10, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(maxPepChargeSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 14;
        gbc.weightx = 0;
        panel.add(createLabel("Minimum Fragment m/z:", "The minimum mz of fragment to consider for library generation"),
                gbc);

        // Initializing libMinFragMzSpinner for -lf_frag_mz_min
        libMinFragMzSpinner = createSpinner(200, 50, 500, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(libMinFragMzSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 15;
        gbc.weightx = 0;
        panel.add(createLabel("Maximum Fragment m/z:", "The maximum mz of fragment to consider for library generation"),
                gbc);

        libMaxFragMzSpinner = createSpinner(1800, 500, 3000, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(libMaxFragMzSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 16;
        gbc.weightx = 0;
        panel.add(createLabel("Maximum Number of Fragment Ions:",
                "The maximum number of fragment ions to consider for library generation"), gbc);

        LibTopNFragIonsSpinner = createSpinner(20, 6, 100, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(LibTopNFragIonsSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 17;
        gbc.weightx = 0;
        panel.add(createLabel("Minimum Number of Fragment Ions:",
                "The minimum number of fragment ions to consider for library generation"), gbc);

        libMinNumFragSpinner = createSpinner(2, 1, 3, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(libMinNumFragSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 18;
        gbc.weightx = 0;
        panel.add(createLabel("Minimum Fragment Ion Number:",
                "The minimum fragment ion number to consider for library generation"), gbc);

        libFragNumMinSpinner = createSpinner(2, 1, 3, 1);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(libFragNumMinSpinner, gbc);

        gbc.gridx = 0;
        gbc.gridy = 19;
        gbc.weightx = 0;
        panel.add(createLabel("Spectral Library Format:", "Spectral library format"), gbc);

        String[] formats = { "DIA-NN", "Skyline", "EncyclopeDIA", "mzSpecLib" };
        libraryFormatCombo = new JComboBox<>(formats);
        styleComboBox(libraryFormatCombo);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(libraryFormatCombo, gbc);

        gbc.gridx = 0;
        gbc.gridy = 20;
        gbc.gridwidth = 2;
        gbc.weighty = 1;
        panel.add(Box.createVerticalGlue(), gbc);

        return panel;
    }

    private JPanel createConsolePanel() {
        JPanel panel = new JPanel(new BorderLayout());
        Color border = lafColor("Component.borderColor", lafColor("Separator.foreground", new Color(128, 128, 128)));
        // Remove outer line border, keep padding
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // 1. Create a header wrapper for the title and the copy button
        JPanel headerWrapper = new JPanel(new BorderLayout());
        headerWrapper.setOpaque(false);
        headerWrapper.setBorder(BorderFactory.createEmptyBorder(0, 0, 8, 0));

        JLabel consoleLabel = new JLabel("[>] Console Output");
        consoleLabel.setFont(new Font("Segoe UI", Font.BOLD, 13));
        headerWrapper.add(consoleLabel, BorderLayout.WEST);

        // 2. Add the Copy button with the same style as other buttons
        JButton copyButton = new JButton("Copy Output");
        styleButton(copyButton);
        copyButton.addActionListener(e -> {
            String content = consoleArea.getText();
            if (content != null && !content.isEmpty()) {
                java.awt.datatransfer.StringSelection selection = new java.awt.datatransfer.StringSelection(content);
                java.awt.Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);

                // Visual feedback: briefly change text
                copyButton.setText("Copied!");
                new javax.swing.Timer(1500, evt -> copyButton.setText("Copy Output")).start();
            }
        });
        headerWrapper.add(copyButton, BorderLayout.EAST);

        panel.add(headerWrapper, BorderLayout.NORTH);

        consoleArea = new JTextArea();
        consoleArea.setEditable(false);
        consoleArea.setLineWrap(false); // OPTIMIZATION: process copious output much faster
        consoleArea.setWrapStyleWord(false);

        consoleScrollPane = new JScrollPane(consoleArea);
        // Restore inner border for the console output box
        consoleScrollPane.setBorder(BorderFactory.createLineBorder(border));
        consoleScrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        panel.add(consoleScrollPane, BorderLayout.CENTER);

        // Apply theme once now (and again on toggle via applyThemeToCustomComponents)
        applyThemeToCustomComponents();

        return panel;
    }

    private JPanel createFooter() {
        JPanel footer = new JPanel(new BorderLayout());
        Color border = lafColor("Component.borderColor", lafColor("Separator.foreground", new Color(128, 128, 128)));
        footer.setBorder(BorderFactory.createMatteBorder(1, 0, 0, 0, border));

        progressBar = new JProgressBar();
        progressBar.setIndeterminate(false);
        progressBar.setStringPainted(true);
        progressBar.setString("Ready");
        progressBar.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        progressBar.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));
        footer.add(progressBar, BorderLayout.NORTH);

        JPanel buttonsPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 15, 15));

        runButton = createPrimaryButton("Run Carafe", ACCENT_COLOR);
        runButton.addActionListener(e -> runCarafe());
        buttonsPanel.add(runButton);

        stopButton = createPrimaryButton("Stop", new Color(231, 76, 60));
        stopButton.setEnabled(false);
        stopButton.addActionListener(e -> stopCarafe());
        buttonsPanel.add(stopButton);

        JButton previewButton = createSecondaryButton("Preview Command");
        previewButton.addActionListener(e -> previewCommand());
        buttonsPanel.add(previewButton);

        JButton clearButton = createSecondaryButton("Clear Console");
        clearButton.addActionListener(e -> consoleArea.setText(""));
        buttonsPanel.add(clearButton);

        JButton helpButton = createSecondaryButton("Help");
        helpButton.addActionListener(e -> showHelp());
        buttonsPanel.add(helpButton);

        footer.add(buttonsPanel, BorderLayout.CENTER);

        JPanel statusBar = new JPanel(new BorderLayout());
        statusBar.setBorder(BorderFactory.createEmptyBorder(5, 15, 5, 15));

        this.statusLabel = new JLabel("Ready | GPU: " + cachedGpuStatus);
        statusLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        statusBar.add(statusLabel, BorderLayout.WEST);

        JLabel memoryLabel = new JLabel("Java: " + System.getProperty("java.version"));
        memoryLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        statusBar.add(memoryLabel, BorderLayout.EAST);

        footer.add(statusBar, BorderLayout.SOUTH);

        return footer;
    }

    private boolean isGPUAvailable(String pyPath) {
        try {
            if (CudaUtils.hasCuda()) {
                return true;
            } else {
                GPUTools gpuTools = new GPUTools();
                if (pyPath != null && !pyPath.isEmpty()) {
                    gpuTools.py_path = pyPath;
                }
                GPUTools.TorchGpuStatus st = gpuTools.checkTorchGpu();
                return st.gpuAvailable;
            }
        } catch (Exception e) {
            return false;
        }
    }

    private void previewCommand() {
        String command = buildCommand();
        JTextArea commandArea = new JTextArea(command);
        commandArea.setFont(new Font("Consolas", Font.PLAIN, 12));
        commandArea.setEditable(false);
        commandArea.setLineWrap(true);
        commandArea.setWrapStyleWord(true);

        JScrollPane scrollPane = new JScrollPane(commandArea);
        scrollPane.setPreferredSize(new Dimension(600, 200));

        int result = JOptionPane.showOptionDialog(this, scrollPane, "Command Preview",
                JOptionPane.DEFAULT_OPTION, JOptionPane.INFORMATION_MESSAGE, null,
                new String[] { "Copy to Clipboard", "Close" }, "Close");

        if (result == 0) {
            java.awt.datatransfer.StringSelection selection = new java.awt.datatransfer.StringSelection(command);
            java.awt.Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);
            JOptionPane.showMessageDialog(this, "Command copied to clipboard!", "Copied",
                    JOptionPane.INFORMATION_MESSAGE);
        }
    }

    private void refreshStatusLabel() {
        if (statusLabel == null)
            return;

        String py = "Not Set";
        try {
            if (pythonPathCombo != null && pythonPathCombo.getSelectedItem() != null) {
                py = pythonPathCombo.getSelectedItem().toString();
            }
        } catch (Exception ignored) {
        }

        statusLabel.setText("Ready | GPU: " + cachedGpuStatus + " | Python: " + py);
    }

    private void updateGpuStatusAsync() {
        if (pythonPathCombo == null)
            return;

        // Capture the path on the EDT
        final String currentPy = (pythonPathCombo.getSelectedItem() != null)
                ? pythonPathCombo.getSelectedItem().toString()
                : "";

        this.cachedGpuStatus = "Checking...";
        refreshStatusLabel();

        new Thread(() -> {
            try {
                // Use the captured path safely in the background
                boolean available = isGPUAvailable(currentPy);
                SwingUtilities.invokeLater(() -> {
                    this.cachedGpuStatus = available ? "Available" : "Not Available";
                    refreshStatusLabel();
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    this.cachedGpuStatus = "Error";
                    refreshStatusLabel();
                });
            }
        }).start();
    }

    // Helper methods for creating styled components

    private JLabel createLabel(String text) {
        JLabel label = new JLabel(text);
        label.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        return label;
    }

    private JLabel createLabel(String text, String toolTip) {
        JLabel label = createLabel(text);
        label.setToolTipText(toolTip);
        return label;
    }

    private JTextField createTextField(String placeholder) {
        JTextField field = new JTextField(10) {
            @Override
            public Dimension getPreferredSize() {
                Dimension d = super.getPreferredSize();
                return new Dimension(100, d.height);
            }
        };
        field.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        field.setToolTipText(placeholder);
        return field;
    }

    private JButton createBrowseButton(JTextField targetField, String description, String[] extensions) {
        JButton button = new JButton("Browse");
        styleButton(button);
        button.addActionListener(e -> {
            FileNameExtensionFilter filter = null;
            if (extensions != null && extensions.length > 0) {
                filter = new FileNameExtensionFilter(description, extensions);
            }
            chooseFile("Select File", JFileChooser.FILES_ONLY, filter, f -> {
                targetField.setText(f.getAbsolutePath());
                prefs.put(PREF_LAST_DIR, f.getParent());
            });
        });
        return button;
    }

    private JButton createFolderButton(JTextField targetField) {
        JButton button = new JButton("Folder");
        styleButton(button);
        button.addActionListener(e -> {
            chooseFile("Select Folder", JFileChooser.DIRECTORIES_ONLY, null, f -> {
                targetField.setText(f.getAbsolutePath());
                prefs.put(PREF_LAST_DIR, f.getAbsolutePath());
            });
        });
        return button;
    }

    private JPanel createPythonBrowseButton() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));
        panel.setOpaque(false);

        JButton browse = new JButton("Browse");
        styleButton(browse);
        browse.addActionListener(e -> {
            setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
            new Thread(() -> {
                try {
                    JFileChooser chooser = new JFileChooser();
                    String defaultDir = System.getProperty("os.name").toLowerCase().contains("windows") ? "C:\\"
                            : "/usr/bin";
                    String lastDir = prefs.get(PREF_LAST_DIR, defaultDir);
                    chooser.setCurrentDirectory(new File(lastDir));
                    chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                    chooser.setDialogTitle("Select Python Executable");
                    if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                        chooser.setFileFilter(
                                new javax.swing.filechooser.FileNameExtensionFilter("Executable Files", "exe"));
                    }
                    SwingUtilities.invokeLater(() -> {
                        setCursor(Cursor.getDefaultCursor());
                        if (chooser.showOpenDialog(CarafeGUI.this) == JFileChooser.APPROVE_OPTION) {
                            File selectedFile = chooser.getSelectedFile();
                            String path = selectedFile.getAbsolutePath();

                            boolean found = false;
                            for (int i = 0; i < pythonPathCombo.getItemCount(); i++) {
                                if (pythonPathCombo.getItemAt(i).equals(path)) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                pythonPathCombo.addItem(path);
                            }
                            pythonPathCombo.setSelectedItem(path);

                            prefs.put(PREF_PYTHON_PATH, path);
                            prefs.put(PREF_LAST_DIR, selectedFile.getParent());
                        }
                    });
                } catch (Exception ex) {
                    SwingUtilities.invokeLater(() -> setCursor(Cursor.getDefaultCursor()));
                    ex.printStackTrace();
                }
            }).start();
        });

        JButton install = new JButton("Install");
        styleButton(install);
        install.addActionListener(e -> {
            install.setEnabled(false);
            browse.setEnabled(false);
            CarafeGUI.this.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));

            new Thread(() -> {
                String home = System.getProperty("user.home");
                java.nio.file.Path installRoot = Paths.get(home, ".carafe");
                try {
                    SwingUtilities.invokeLater(() -> {
                        if (tabbedPane != null)
                            tabbedPane.setSelectedIndex(Math.max(0, tabbedPane.getTabCount() - 1));
                        progressBar.setIndeterminate(true);
                        progressBar.setString("Python installation...");
                        logToConsole("\n[INSTALL] Python installation started...\n");
                        consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
                    });

                    java.nio.file.Path logFile = installRoot.resolve("logs").resolve("install.log");
                    AtomicBoolean installDone = new AtomicBoolean(false);
                    Thread tailer = new Thread(() -> {
                        try {
                            while (!installDone.get() && !java.nio.file.Files.exists(logFile)) {
                                Thread.sleep(200);
                            }
                            if (!java.nio.file.Files.exists(logFile))
                                return;
                            try (RandomAccessFile raf = new RandomAccessFile(logFile.toFile(), "r")) {
                                long pointer = 0;
                                while (!installDone.get()) {
                                    long len = raf.length();
                                    if (len > pointer) {
                                        raf.seek(pointer);
                                        String line;
                                        while ((line = raf.readLine()) != null) {
                                            final String decoded = new String(line.getBytes("ISO-8859-1"),
                                                    StandardCharsets.UTF_8);
                                            SwingUtilities.invokeLater(() -> {
                                                logToConsole(decoded + "\n");
                                                consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
                                            });
                                        }
                                        pointer = raf.getFilePointer();
                                    }
                                    Thread.sleep(200);
                                }
                                long len = raf.length();
                                if (len > pointer) {
                                    raf.seek(pointer);
                                    String line;
                                    while ((line = raf.readLine()) != null) {
                                        final String decoded = new String(line.getBytes("ISO-8859-1"),
                                                StandardCharsets.UTF_8);
                                        SwingUtilities.invokeLater(() -> {
                                            logToConsole(decoded + "\n");
                                            consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
                                        });
                                    }
                                }
                            } catch (java.io.FileNotFoundException fnf) {
                                // ignore
                            } catch (IOException ex) {
                                throw new RuntimeException(ex);
                            }
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                        }
                    });
                    tailer.setDaemon(true);
                    tailer.start();

                    String py_path = PyInstaller.installAll(installRoot);
                    installDone.set(true);
                    try {
                        tailer.join(500);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }

                    final String installedPath = py_path;
                    SwingUtilities.invokeLater(() -> {
                        logToConsole("[INSTALL] Completed. Python installed at: " + installedPath + "\n");
                        consoleArea.setCaretPosition(consoleArea.getDocument().getLength());

                        boolean found = false;
                        for (int i = 0; i < pythonPathCombo.getItemCount(); i++) {
                            if (pythonPathCombo.getItemAt(i).equals(installedPath)) {
                                found = true;
                                break;
                            }
                        }
                        if (!found)
                            pythonPathCombo.addItem(installedPath);
                        pythonPathCombo.setSelectedItem(installedPath);
                        prefs.put(PREF_PYTHON_PATH, installedPath);

                        refreshStatusLabel();

                        JOptionPane.showMessageDialog(CarafeGUI.this,
                                "Python installed: " + installedPath,
                                "Install Complete",
                                JOptionPane.INFORMATION_MESSAGE);
                    });
                } catch (Exception ex) {
                    final String msg = ex.getMessage() == null ? ex.toString() : ex.getMessage();
                    SwingUtilities.invokeLater(() -> {
                        logToConsole("[INSTALL] Failed: " + msg + "\n");
                        consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
                        JOptionPane.showMessageDialog(CarafeGUI.this,
                                "Python installation failed:\n" + msg,
                                "Install Error",
                                JOptionPane.ERROR_MESSAGE);
                    });
                } finally {
                    SwingUtilities.invokeLater(() -> {
                        install.setEnabled(true);
                        browse.setEnabled(true);
                        CarafeGUI.this.setCursor(Cursor.getDefaultCursor());
                        progressBar.setIndeterminate(false);
                        progressBar.setString("Ready");
                    });
                }
            }).start();
        });

        panel.add(browse);
        panel.add(install);
        return panel;
    }

    private JComboBox<String> createPythonComboBox() {
        JComboBox<String> combo = new JComboBox<>();
        combo.setEditable(true);
        combo.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        combo.setToolTipText("Select a detected Python or enter a custom path");

        final boolean isWindows = System.getProperty("os.name").toLowerCase().contains("windows");
        String pythonPrototype = isWindows ? "C:\\Python39\\python.exe" : "/usr/bin/python3";
        combo.setPrototypeDisplayValue(pythonPrototype);

        // java.util.List<String> pythonPaths = detectPythonInstallations();
        java.util.List<String> pythonPaths = new java.util.ArrayList<>();
        for (String path : pythonPaths) {
            combo.addItem(path);
        }

        String savedPath = prefs.get(PREF_PYTHON_PATH, "");
        if (!savedPath.isEmpty()) {
            boolean found = false;
            for (int i = 0; i < combo.getItemCount(); i++) {
                if (combo.getItemAt(i).equals(savedPath)) {
                    combo.setSelectedIndex(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                combo.insertItemAt(savedPath, 0);
                combo.setSelectedIndex(0);
            }
        } else if (combo.getItemCount() > 0) {
            combo.setSelectedIndex(0);
        }

        combo.addActionListener(e -> {
            Object selected = combo.getSelectedItem();
            if (selected != null) {
                prefs.put(PREF_PYTHON_PATH, selected.toString());
                // Trigger background check instead of slow EDT check
                updateGpuStatusAsync();
            }
        });

        return combo;
    }

    private java.util.List<String> detectPythonInstallations() {
        java.util.List<String> pythonPaths = new java.util.ArrayList<>();
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("windows");

        if (isWindows) {
            String[] windowsPaths = {
                    System.getenv("LOCALAPPDATA") + "\\Programs\\Python",
                    System.getenv("PROGRAMFILES") + "\\Python",
                    System.getenv("PROGRAMFILES(X86)") + "\\Python",
                    System.getenv("USERPROFILE") + "\\AppData\\Local\\Programs\\Python",
                    System.getenv("USERPROFILE") + "\\anaconda3",
                    System.getenv("USERPROFILE") + "\\miniconda3",
                    System.getenv("USERPROFILE") + "\\.conda\\envs",
                    "C:\\Python",
                    "C:\\Anaconda3",
                    "C:\\Miniconda3"
            };

            for (String basePath : windowsPaths) {
                if (basePath == null)
                    continue;
                File baseDir = new File(basePath);
                if (baseDir.exists() && baseDir.isDirectory()) {
                    File pythonExe = new File(baseDir, "python.exe");
                    if (pythonExe.exists()) {
                        pythonPaths.add(pythonExe.getAbsolutePath());
                    }
                    File[] subDirs = baseDir.listFiles(File::isDirectory);
                    if (subDirs != null) {
                        for (File subDir : subDirs) {
                            pythonExe = new File(subDir, "python.exe");
                            if (pythonExe.exists()) {
                                pythonPaths.add(pythonExe.getAbsolutePath());
                            }
                        }
                    }
                }
            }

            try {
                ProcessBuilder pb = new ProcessBuilder("where", "python");
                pb.redirectErrorStream(true);
                Process p = pb.start();
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        line = line.trim();
                        if (!line.isEmpty() && new File(line).exists() && !pythonPaths.contains(line)) {
                            pythonPaths.add(line);
                        }
                    }
                }
            } catch (Exception ignored) {
            }

            detectPythonFromRegistry(pythonPaths);

        } else {
            String[] unixPaths = {
                    "/usr/bin/python3",
                    "/usr/bin/python",
                    "/usr/local/bin/python3",
                    "/usr/local/bin/python",
                    System.getenv("HOME") + "/anaconda3/bin/python",
                    System.getenv("HOME") + "/miniconda3/bin/python",
                    System.getenv("HOME") + "/.conda/envs",
                    "/opt/anaconda3/bin/python",
                    "/opt/miniconda3/bin/python"
            };

            for (String path : unixPaths) {
                if (path == null)
                    continue;
                File file = new File(path);
                if (file.exists() && file.canExecute()) {
                    pythonPaths.add(path);
                } else if (file.isDirectory()) {
                    File[] envDirs = file.listFiles(File::isDirectory);
                    if (envDirs != null) {
                        for (File envDir : envDirs) {
                            File pythonExe = new File(envDir, "bin/python");
                            if (pythonExe.exists() && pythonExe.canExecute()) {
                                pythonPaths.add(pythonExe.getAbsolutePath());
                            }
                        }
                    }
                }
            }

            try {
                ProcessBuilder pb = new ProcessBuilder("which", "-a", "python3");
                pb.redirectErrorStream(true);
                Process p = pb.start();
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        line = line.trim();
                        if (!line.isEmpty() && new File(line).exists() && !pythonPaths.contains(line)) {
                            pythonPaths.add(line);
                        }
                    }
                }
            } catch (Exception ignored) {
            }

            try {
                ProcessBuilder pb = new ProcessBuilder("which", "-a", "python");
                pb.redirectErrorStream(true);
                Process p = pb.start();
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        line = line.trim();
                        if (!line.isEmpty() && new File(line).exists() && !pythonPaths.contains(line)) {
                            pythonPaths.add(line);
                        }
                    }
                }
            } catch (Exception ignored) {
            }
        }

        return pythonPaths;
    }

    private void detectPythonFromRegistry(java.util.List<String> pythonPaths) {
        String[] registryKeys = {
                "HKEY_CURRENT_USER\\Software\\Python\\PythonCore",
                "HKEY_LOCAL_MACHINE\\Software\\Python\\PythonCore",
                "HKEY_LOCAL_MACHINE\\Software\\Wow6432Node\\Python\\PythonCore",
                "HKEY_CURRENT_USER\\Software\\Python\\ContinuumAnalytics",
                "HKEY_LOCAL_MACHINE\\Software\\Python\\ContinuumAnalytics"
        };

        for (String baseKey : registryKeys) {
            try {
                ProcessBuilder pb = new ProcessBuilder("reg", "query", baseKey);
                pb.redirectErrorStream(true);
                Process p = pb.start();
                java.util.List<String> versionKeys = new java.util.ArrayList<>();

                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        line = line.trim();
                        if (line.startsWith("HKEY_") && !line.equals(baseKey)) {
                            versionKeys.add(line);
                        }
                    }
                }
                p.waitFor();

                for (String versionKey : versionKeys) {
                    String installPathKey = versionKey + "\\InstallPath";
                    try {
                        ProcessBuilder pb2 = new ProcessBuilder("reg", "query", installPathKey, "/ve");
                        pb2.redirectErrorStream(true);
                        Process p2 = pb2.start();

                        try (BufferedReader reader = new BufferedReader(new InputStreamReader(p2.getInputStream()))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                if (line.contains("REG_SZ")) {
                                    int regSzIndex = line.indexOf("REG_SZ");
                                    if (regSzIndex != -1) {
                                        String installPath = line.substring(regSzIndex + 6).trim();
                                        File pythonExe = new File(installPath, "python.exe");
                                        if (pythonExe.exists() && !pythonPaths.contains(pythonExe.getAbsolutePath())) {
                                            pythonPaths.add(pythonExe.getAbsolutePath());
                                        }
                                    }
                                }
                            }
                        }
                        p2.waitFor();
                    } catch (Exception ignored) {
                    }
                }
            } catch (Exception ignored) {
            }
        }
    }

    private JButton createDiannBrowseButton() {
        JButton button = new JButton("Browse");
        styleButton(button);
        button.addActionListener(e -> {
            setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
            new Thread(() -> {
                try {
                    JFileChooser chooser = new JFileChooser();
                    String defaultDir = System.getProperty("os.name").toLowerCase().contains("windows") ? "C:\\"
                            : "/usr/bin";
                    String lastDir = prefs.get(PREF_LAST_DIR, defaultDir);
                    chooser.setCurrentDirectory(new File(lastDir));
                    chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                    chooser.setDialogTitle("Select DIA-NN Executable");
                    if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                        chooser.setFileFilter(
                                new javax.swing.filechooser.FileNameExtensionFilter("Executable Files", "exe"));
                    }
                    SwingUtilities.invokeLater(() -> {
                        setCursor(Cursor.getDefaultCursor());
                        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                            File selectedFile = chooser.getSelectedFile();
                            String path = selectedFile.getAbsolutePath();

                            boolean found = false;
                            for (int i = 0; i < diannPathCombo.getItemCount(); i++) {
                                if (diannPathCombo.getItemAt(i).equals(path)) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                diannPathCombo.addItem(path);
                            }
                            diannPathCombo.setSelectedItem(path);
                            prefs.put(PREF_DIANN_PATH, path);
                            prefs.put(PREF_LAST_DIR, selectedFile.getParent());
                        }
                    });
                } catch (Exception ex) {
                    SwingUtilities.invokeLater(() -> setCursor(Cursor.getDefaultCursor()));
                    ex.printStackTrace();
                }
            }).start();
        });
        return button;
    }

    private JComboBox<String> createDiannComboBox() {
        JComboBox<String> combo = new JComboBox<>();
        combo.setEditable(true);
        combo.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        combo.setToolTipText("Select a detected DIA-NN or enter a custom path");

        String diannPrototype = System.getProperty("os.name").toLowerCase().contains("windows")
                ? "C:\\DIA-NN\\diann.exe"
                : "/usr/local/bin/diann";
        combo.setPrototypeDisplayValue(diannPrototype);

        java.util.List<String> diannPaths = detectDiannInstallations();
        for (String path : diannPaths) {
            combo.addItem(path);
        }

        String savedPath = prefs.get(PREF_DIANN_PATH, "");
        if (!savedPath.isEmpty()) {
            boolean found = false;
            for (int i = 0; i < combo.getItemCount(); i++) {
                if (combo.getItemAt(i).equals(savedPath)) {
                    combo.setSelectedIndex(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                combo.insertItemAt(savedPath, 0);
                combo.setSelectedIndex(0);
            }
        } else if (combo.getItemCount() > 0) {
            combo.setSelectedIndex(0);
        }

        combo.addActionListener(e -> {
            Object selected = combo.getSelectedItem();
            if (selected != null) {
                prefs.put(PREF_DIANN_PATH, selected.toString());
            }
        });

        return combo;
    }

    private java.util.List<String> detectDiannInstallations() {
        java.util.List<String> diannPaths = new java.util.ArrayList<>();
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("windows");

        if (isWindows) {
            String[] windowsPaths = {
                    System.getenv("PROGRAMFILES") + "\\DIA-NN",
                    System.getenv("PROGRAMFILES(X86)") + "\\DIA-NN",
                    System.getenv("LOCALAPPDATA") + "\\DIA-NN",
                    System.getenv("USERPROFILE") + "\\DIA-NN",
                    "C:\\DIA-NN",
                    "C:\\Program Files\\DIA-NN",
                    "C:\\Program Files (x86)\\DIA-NN",
                    "D:\\DIA-NN",
                    "D:\\Program Files\\DIA-NN",
                    "D:\\Program Files (x86)\\DIA-NN",
                    "E:\\DIA-NN",
                    "E:\\Program Files\\DIA-NN",
                    "E:\\Program Files (x86)\\DIA-NN"
            };

            for (String basePath : windowsPaths) {
                File baseDir = new File(basePath);
                if (baseDir.exists() && baseDir.isDirectory()) {
                    File diannExe = new File(baseDir, "diann.exe");
                    if (diannExe.exists() && !diannPaths.contains(diannExe.getAbsolutePath())) {
                        diannPaths.add(diannExe.getAbsolutePath());
                    }
                    File[] subDirs = baseDir.listFiles(File::isDirectory);
                    if (subDirs != null) {
                        for (File subDir : subDirs) {
                            diannExe = new File(subDir, "diann.exe");
                            if (diannExe.exists() && !diannPaths.contains(diannExe.getAbsolutePath())) {
                                diannPaths.add(diannExe.getAbsolutePath());
                            }
                        }
                    }
                }
            }

            // Special check for user request: Root:\*\DIA-NN pattern
            // Scans all available root drives (C:\, D:\, etc.) for a "DIA-NN" folder one
            // level deep
            File[] roots = File.listRoots();
            if (roots != null) {
                for (File root : roots) {
                    if (root.exists() && root.isDirectory()) {
                        File[] subDirs = root.listFiles(File::isDirectory);
                        if (subDirs != null) {
                            for (File dir : subDirs) {
                                File diannSubDir = new File(dir, "DIA-NN");
                                if (diannSubDir.exists() && diannSubDir.isDirectory()) {
                                    // Check for C:\Something\DIA-NN\diann.exe
                                    File diannExe = new File(diannSubDir, "diann.exe");
                                    if (diannExe.exists() && !diannPaths.contains(diannExe.getAbsolutePath())) {
                                        diannPaths.add(diannExe.getAbsolutePath());
                                    }

                                    // Check for C:\Something\DIA-NN\1.8.1\diann.exe (versioned subdirs)
                                    File[] deepDirs = diannSubDir.listFiles(File::isDirectory);
                                    if (deepDirs != null) {
                                        for (File deep : deepDirs) {
                                            File deepExe = new File(deep, "diann.exe");
                                            if (deepExe.exists() && !diannPaths.contains(deepExe.getAbsolutePath())) {
                                                diannPaths.add(deepExe.getAbsolutePath());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            }

            try {
                ProcessBuilder pb = new ProcessBuilder("where", "diann");
                pb.redirectErrorStream(true);
                Process p = pb.start();
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        line = line.trim();
                        if (!line.isEmpty() && new File(line).exists() && !diannPaths.contains(line)) {
                            diannPaths.add(line);
                        }
                    }
                }
            } catch (Exception ignored) {
            }

        } else

        {
            String[] unixPaths = {
                    "/usr/local/bin/diann",
                    "/usr/bin/diann",
                    System.getenv("HOME") + "/DIA-NN/diann",
                    "/opt/DIA-NN/diann"
            };

            for (String path : unixPaths) {
                File file = new File(path);
                if (file.exists() && file.canExecute()) {
                    diannPaths.add(path);
                }
            }

            try {
                ProcessBuilder pb = new ProcessBuilder("which", "diann");
                pb.redirectErrorStream(true);
                Process p = pb.start();
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        line = line.trim();
                        if (!line.isEmpty() && new File(line).exists() && !diannPaths.contains(line)) {
                            diannPaths.add(line);
                        }
                    }
                }
            } catch (Exception ignored) {
            }
        }

        if (diannPaths.isEmpty()) {
            // diannPaths.add(isWindows ? "diann.exe" : "diann");
        }
        return diannPaths;
    }

    private void updateMsConvertVisibility() {
        if (msConvertExeRowComponents == null)
            return;

        boolean show = false;

        // 1. Check Multi-Select List
        if (trainMsFiles != null && !trainMsFiles.isEmpty()) {
            for (String f : trainMsFiles) {
                if (f.toLowerCase().endsWith(".raw")) {
                    show = true;
                    break;
                }
            }
        }
        // 2. Check Single File / Folder Text
        else if (trainMsFileField != null && trainMsFileField.isVisible()) {
            String text = trainMsFileField.getText().trim();
            if (!text.isEmpty()) {
                if (text.toLowerCase().endsWith(".raw")) {
                    show = true;
                } else {
                    File file = new File(text);
                    if (file.isDirectory()) {
                        File[] rawFiles = file.listFiles((dir, name) -> name.toLowerCase().endsWith(".raw"));
                        if (rawFiles != null && rawFiles.length > 0) {
                            show = true;
                        }
                    }
                }
            }
        }

        setVisible(msConvertExeRowComponents, show);
        if (inputFieldsPanel != null) {
            inputFieldsPanel.revalidate();
            inputFieldsPanel.repaint();
        }
    }

    private JComboBox<String> createMsConvertComboBox() {
        JComboBox<String> combo = new JComboBox<>();
        combo.setEditable(true);
        combo.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        combo.setToolTipText("Select ProteoWizard MSConvert executable");

        String msConvertPrototype = "C:\\Program Files\\ProteoWizard\\ProteoWizard 3.0.x\\msconvert.exe";
        combo.setPrototypeDisplayValue(msConvertPrototype);

        java.util.List<String> paths = detectMsConvertInstallations();
        for (String path : paths) {
            combo.addItem(path);
        }

        String savedPath = prefs.get(PREF_MSCONVERT_PATH, "");
        if (!savedPath.isEmpty()) {
            boolean found = false;
            for (int i = 0; i < combo.getItemCount(); i++) {
                if (combo.getItemAt(i).equals(savedPath)) {
                    combo.setSelectedIndex(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                combo.insertItemAt(savedPath, 0);
                combo.setSelectedIndex(0);
            }
        } else if (combo.getItemCount() > 0) {
            combo.setSelectedIndex(0);
        }

        combo.addActionListener(e -> {
            Object selected = combo.getSelectedItem();
            if (selected != null) {
                prefs.put(PREF_MSCONVERT_PATH, selected.toString());
            }
        });
        return combo;
    }

    private JButton createMsConvertBrowseButton() {
        JButton button = new JButton("Browse");
        styleButton(button);
        button.addActionListener(e -> {
            setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
            new Thread(() -> {
                try {
                    JFileChooser chooser = new JFileChooser();
                    String lastDir = prefs.get(PREF_LAST_DIR, "C:\\");
                    chooser.setCurrentDirectory(new File(lastDir));
                    chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                    chooser.setDialogTitle("Select MSConvert Executable");
                    chooser.setFileFilter(new FileNameExtensionFilter("Executable Files", "exe"));

                    SwingUtilities.invokeLater(() -> {
                        setCursor(Cursor.getDefaultCursor());
                        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                            File f = chooser.getSelectedFile();
                            String path = f.getAbsolutePath();

                            boolean found = false;
                            for (int i = 0; i < msConvertPathCombo.getItemCount(); i++) {
                                if (msConvertPathCombo.getItemAt(i).equals(path)) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                msConvertPathCombo.addItem(path);
                            }
                            msConvertPathCombo.setSelectedItem(path);
                            prefs.put(PREF_MSCONVERT_PATH, path);
                            prefs.put(PREF_LAST_DIR, f.getParent());
                        }
                    });
                } catch (Exception ex) {
                    SwingUtilities.invokeLater(() -> setCursor(Cursor.getDefaultCursor()));
                    ex.printStackTrace();
                }
            }).start();
        });
        return button;
    }

    private java.util.List<String> detectMsConvertInstallations() {
        java.util.List<String> paths = new java.util.ArrayList<>();
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("windows");
        if (!isWindows)
            return paths; // MSConvert is typically Windows only

        // Common locations
        String[] bases = {
                System.getenv("PROGRAMFILES"),
                System.getenv("PROGRAMFILES(X86)"),
                System.getenv("LOCALAPPDATA"),
                "C:\\ProteoWizard"
        };

        for (String base : bases) {
            if (base == null)
                continue;
            File dir = new File(base);
            if (!dir.exists())
                continue;

            // Look for ProteoWizard subfolders
            File[] subdirs = dir.listFiles((d, name) -> name.toLowerCase().contains("proteowizard"));
            if (subdirs != null) {
                for (File sub : subdirs) {
                    File exe = new File(sub, "msconvert.exe");
                    if (exe.exists())
                        paths.add(exe.getAbsolutePath());
                }
            }
        }

        // Try `where msconvert`
        try {
            ProcessBuilder pb = new ProcessBuilder("where", "msconvert");
            pb.redirectErrorStream(true);
            Process p = pb.start();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty() && new File(line).exists() && !paths.contains(line)) {
                        paths.add(line);
                    }
                }
            }
        } catch (Exception ignored) {
        }

        // Try registry query for "Open with MSConvertGUI" (HKCU and HKLM)
        String[] regRoots = { "HKCU", "HKLM" };
        for (String root : regRoots) {
            try {
                ProcessBuilder pb = new ProcessBuilder("reg", "query",
                        root + "\\Software\\Classes\\*\\shell\\Open with MSConvertGUI\\command", "/ve");
                pb.redirectErrorStream(true);
                Process p = pb.start();
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        // Output format usually: (Default) REG_SZ "C:\Path\To\MSConvertGUI.exe" "%1"
                        if (line.contains("MSConvertGUI.exe")) {
                            // Extract path between quotes
                            int start = line.indexOf('"');
                            int end = line.lastIndexOf("MSConvertGUI.exe");
                            if (start >= 0 && end > start) {
                                String exePath = line.substring(start + 1, end + 16); // +16 for "MSConvertGUI.exe"
                                                                                      // length
                                File msConvertGui = new File(exePath);
                                if (msConvertGui.exists()) {
                                    File msConvert = new File(msConvertGui.getParent(), "msconvert.exe");
                                    if (msConvert.exists() && !paths.contains(msConvert.getAbsolutePath())) {
                                        paths.add(msConvert.getAbsolutePath());
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (Exception ignored) {
            }
        }

        return paths;
    }

    private void styleButton(JButton button) {
        button.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));

        button.setMargin(new Insets(6, 12, 6, 12));

        button.putClientProperty("JButton.buttonType", "roundRect");

        button.putClientProperty("JButton.hoverBackground", UIManager.getColor("Button.hoverBackground"));
    }

    private void styleSecondaryButton(JButton button) {
        button.putClientProperty("carafe.role", "secondary");
        button.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        Color border = lafColor("Component.borderColor", lafColor("Separator.foreground", new Color(128, 128, 128)));
        button.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(border),
                BorderFactory.createEmptyBorder(10, 20, 10, 20)));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
    }

    private JButton createPrimaryButton(String text, Color color) {
        JButton button = new JButton(text);
        button.setFont(new Font("Segoe UI", Font.BOLD, 14));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        button.setMargin(new Insets(12, 30, 12, 30));

        button.putClientProperty("JButton.buttonType", "roundRect");
        button.putClientProperty("JButton.background", color);
        button.putClientProperty("JButton.foreground", Color.WHITE);

        return button;
    }

    private JButton createSecondaryButton(String text) {
        JButton button = new JButton(text);
        button.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));

        button.setMargin(new Insets(10, 20, 10, 20));
        button.putClientProperty("JButton.buttonType", "roundRect");

        return button;
    }

    private void styleComboBox(JComboBox<?> combo) {
        combo.setFont(new Font("Segoe UI", Font.PLAIN, 13));
    }

    private JSpinner createSpinner(int value, int min, int max, int step) {
        JSpinner spinner = new JSpinner(new SpinnerNumberModel(value, min, max, step));
        spinner.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        ((JSpinner.DefaultEditor) spinner.getEditor()).getTextField().setColumns(5);
        Dimension prefSize = spinner.getPreferredSize();
        spinner.setPreferredSize(new Dimension(prefSize.width, COMPONENT_HEIGHT));
        spinner.setMinimumSize(new Dimension(60, COMPONENT_HEIGHT));
        return spinner;
    }

    private JSpinner createDoubleSpinner(double value, double min, double max, double step) {
        JSpinner spinner = new JSpinner(new SpinnerNumberModel(value, min, max, step));
        spinner.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        JSpinner.NumberEditor editor = new JSpinner.NumberEditor(spinner, "0.000");
        spinner.setEditor(editor);
        Dimension prefSize = spinner.getPreferredSize();
        spinner.setPreferredSize(new Dimension(prefSize.width, COMPONENT_HEIGHT));
        spinner.setMinimumSize(new Dimension(60, COMPONENT_HEIGHT));
        return spinner;
    }

    private JCheckBox createCheckBox(String text, boolean selected) {
        JCheckBox checkbox = new JCheckBox(text, selected);
        checkbox.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        return checkbox;
    }

    private JPanel createInfoCard(String title, String content) {
        JPanel card = new JPanel(new BorderLayout(0, 8));
        card.setOpaque(true);
        card.setBorder(BorderFactory.createEmptyBorder(16, 16, 16, 16));

        JLabel titleLabel = new JLabel(title);
        titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 13));
        titleLabel.setBorder(BorderFactory.createEmptyBorder(0, 0, 6, 0));
        card.add(titleLabel, BorderLayout.NORTH);

        JTextArea contentArea = new JTextArea(content) {
            @Override
            public Dimension getPreferredSize() {
                Dimension d = super.getPreferredSize();
                return new Dimension(200, d.height);
            }
        };
        contentArea.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        contentArea.setEditable(false);
        contentArea.setLineWrap(true);
        contentArea.setWrapStyleWord(true);

        contentArea.setOpaque(false);
        contentArea.setBorder(null);

        card.add(contentArea, BorderLayout.CENTER);

        InfoCardRef ref = new InfoCardRef(card, titleLabel, contentArea);
        infoCards.add(ref);
        updateInfoCardTheme(ref);

        return card;
    }

    private static class InfoCardRef {
        final JPanel card;
        final JLabel titleLabel;
        final JTextArea contentArea;

        InfoCardRef(JPanel card, JLabel titleLabel, JTextArea contentArea) {
            this.card = card;
            this.titleLabel = titleLabel;
            this.contentArea = contentArea;
        }
    }

    // Action methods

    private String buildCommand() {
        int wf = workflowCombo.getSelectedIndex();
        if (wf == 1)
            return buildCarafeCommand().cmd;
        return "This workflow runs chained commands. Click Run Carafe to execute.";
    }

    private CmdTask buildCarafeCommand() {
        return buildCarafeCommand(null);
    }

    private CmdTask buildCarafeCommand(String trainMsFileOverride) {
        List<String> commandArgs = new ArrayList<>();
        StringBuilder cmd = new StringBuilder();
        boolean exe_launch = false;
        String javaExec = getJavaExecutable();
        if (javaExec.endsWith("java.exe") || javaExec.endsWith("java")) {
            // use as is
        } else if (javaExec.endsWith("Carafe.exe")) {
            // it is likely launched from bundled Carafe.exe
            File javaFile = new File(javaExec);
            // navigate to ../runtime/bin/java.exe
            exe_launch = true;
        } else {
            // fallback to "java" in PATH
            javaExec = "java";
        }
        commandArgs.add(javaExec);

        // This is not needed
        if (javaExec.contains(" ")) {
            javaExec = '"' + javaExec + '"';
        }
        // get the computer memory and set Xmx accordingly
        int memory_use = (int) Math.ceil(GenericUtils.get_system_memory_available() * 0.8);

        if (exe_launch) {
            cmd.append(javaExec).append(" ");
        } else {
            cmd.append(javaExec).append(" -Xmx").append(memory_use).append("G ");
            commandArgs.add("-Xmx" + memory_use + "G");
            int javaVersion = GenericUtils.getJavaMajorVersion();
            if (javaVersion >= 18 && javaVersion <= 23) {
                cmd.append("-Djava.security.manager=allow ");
            }
            commandArgs.add("-jar");
            cmd.append("-jar ");
            String carafeJarPath = getCarafeJarPath();
            commandArgs.add(carafeJarPath);
            cmd.append(carafeJarPath).append(" ");
        }

        // Carafe additional arguments
        String additionalOptions = carafeAdditionalOptionsField.getText().trim();
        ArrayList<String> additionalOptionList = new ArrayList<>();
        // store the index of the additional options which are present through the GUI
        ArrayList<Integer> additionalOptionInGuiList = new ArrayList<>();
        if (!additionalOptions.isEmpty()) {
            String[] additional_options = Commandline.translateCommandline(additionalOptions);
            Collections.addAll(additionalOptionList, additional_options);
        }

        String libraryDb = libraryDbFileField.getText().trim();
        if (!libraryDb.isEmpty()) {
            // cmd.append("-db \"").append(libraryDb).append("\" ");
            commandArgs.add("-db");
            commandArgs.add(libraryDb);
            // commandArgs.add("\"" + libraryDb + "\"");
        }

        String diannReport = diannReportFileField.getText().trim();
        if (!diannReport.isEmpty()) {
            // cmd.append("-i \"").append(diannReport).append("\" ");
            commandArgs.add("-i");
            commandArgs.add(diannReport);
            // commandArgs.add("\"" + diannReport + "\"");
        }

        // Use override if provided, otherwise check trainMsFiles list, then text field
        String trainMsFile;
        if (trainMsFileOverride != null) {
            trainMsFile = trainMsFileOverride;
        } else if (trainMsFiles != null && !trainMsFiles.isEmpty()) {
            // Multi-file selection: use the parent folder of the first file
            if (trainMsFiles.size() >= 2) {
                trainMsFile = new File(trainMsFiles.getFirst()).getParent();
            } else {
                trainMsFile = trainMsFiles.getFirst();
            }
        } else {
            // Single file or folder path from text field
            trainMsFile = trainMsFileField.getText().trim();
        }
        if (!trainMsFile.isEmpty()) {
            // cmd.append("-ms \"").append(trainMsFile).append("\" ");
            commandArgs.add("-ms");
            commandArgs.add(trainMsFile);
            // commandArgs.add("\"" + trainMsFile + "\"");
        }

        String outDir = outputDirField.getText().trim();
        if (!outDir.isEmpty()) {
            carafe_library_directory = outDir + File.separator + "carafe_library";
            // cmd.append("-o \"").append(carafe_library_directory).append("\" ");
            commandArgs.add("-o");
            commandArgs.add(carafe_library_directory);
            // commandArgs.add("\"" + carafe_library_directory + "\"");
        }

        // cmd.append("-fdr ").append(fdrSpinner.getValue()).append(" ");
        commandArgs.add("-fdr");
        commandArgs.add(fdrSpinner.getValue().toString());
        // cmd.append("-ptm_site_prob ").append(ptmSiteProbSpinner.getValue()).append("
        // ");
        commandArgs.add("-ptm_site_prob");
        commandArgs.add(ptmSiteProbSpinner.getValue().toString());
        // cmd.append("-ptm_site_qvalue
        // ").append(ptmSiteQvalueSpinner.getValue()).append(" ");
        commandArgs.add("-ptm_site_qvalue");
        commandArgs.add(ptmSiteQvalueSpinner.getValue().toString());
        // cmd.append("-itol ").append(fragTolSpinner.getValue()).append(" ");
        commandArgs.add("-itol");
        commandArgs.add(fragTolSpinner.getValue().toString());
        // cmd.append("-itolu ").append(fragTolUnitCombo.getSelectedItem()).append(" ");
        commandArgs.add("-itolu");
        commandArgs.add(fragTolUnitCombo.getSelectedItem().toString());
        // if (refineBoundaryCheckbox.isSelected()) cmd.append("-rf ");
        if (refineBoundaryCheckbox.isSelected()) {
            commandArgs.add("-rf");
        }

        String rfRtWin = rtPeakWindowField.getText().trim();
        if (rfRtWin.isEmpty()) {
            commandArgs.add("-rf_rt_win");
            commandArgs.add("auto");
        } else {
            commandArgs.add("-rf_rt_win");
            commandArgs.add(rfRtWin);
        }
        // cmd.append("-cor ").append(xicCorSpinner.getValue()).append(" ");
        commandArgs.add("-cor");
        commandArgs.add(xicCorSpinner.getValue().toString());
        commandArgs.add("-min_mz");
        commandArgs.add(minFragMzSpinner.getValue().toString());

        // -n_ion_min
        commandArgs.add("-n_ion_min");
        commandArgs.add(nIonMinSpinner.getValue().toString());

        // -c_ion_min
        commandArgs.add("-c_ion_min");
        commandArgs.add(cIonMinSpinner.getValue().toString());

        // cmd.append("-mode ").append(modeCombo.getSelectedItem()).append(" ");
        commandArgs.add("-mode");
        commandArgs.add(modeCombo.getSelectedItem().toString());
        String nce = nceField.getText().trim();
        if (!nce.isEmpty()) {
            if (!nce.equalsIgnoreCase("auto")) {
                // cmd.append("-nce ").append(nce).append(" ");
                commandArgs.add("-nce");
                commandArgs.add(nce);
            }
        }
        Object msSel = msInstrumentField.getSelectedItem();
        String msInstrument = msSel == null ? "" : msSel.toString().trim();
        if (!msInstrument.isEmpty()) {
            if (!msInstrument.equalsIgnoreCase("auto")) {
                // cmd.append("-ms_instrument ").append(msInstrument).append(" ");
                commandArgs.add("-ms_instrument");
                commandArgs.add(msInstrument);
            }
        }

        Object deviceSel = deviceCombo.getSelectedItem();
        String device = deviceSel == null ? "auto" : deviceSel.toString().trim();
        if (device.equalsIgnoreCase("auto")) {
            boolean available = "Available".equals(cachedGpuStatus);
            // cmd.append("-device ").append(available ? "gpu" : "cpu").append(" ");
            commandArgs.add("-device");
            commandArgs.add(available ? "gpu" : "cpu");
        } else {
            // cmd.append("-device ").append(device).append(" ");
            commandArgs.add("-device");
            commandArgs.add(device);
        }

        String enzyme = ((String) enzymeCombo.getSelectedItem()).split(":")[0];
        // cmd.append("-enzyme ").append(enzyme).append(" ");
        commandArgs.add("-enzyme");
        commandArgs.add(enzyme);
        // cmd.append("-miss_c ").append(missCleavageSpinner.getValue()).append(" ");
        commandArgs.add("-miss_c");
        commandArgs.add(missCleavageSpinner.getValue().toString());

        String fixModSelected = fixModSelectedField.getText().trim();
        if (!fixModSelected.isEmpty()) {
            // cmd.append("-fixMod ").append(fixModSelected).append(" ");
            commandArgs.add("-fixMod");
            commandArgs.add(fixModSelected);
        }

        String varModSelected = varModSelectedField.getText().trim();
        if (!varModSelected.isEmpty()) {
            // cmd.append("-varMod ").append(varModSelected).append(" ");
            commandArgs.add("-varMod");
            commandArgs.add(varModSelected);
        }

        // cmd.append("-maxVar ").append(maxVarSpinner.getValue()).append(" ");
        commandArgs.add("-maxVar");
        commandArgs.add(maxVarSpinner.getValue().toString());
        if (clipNmCheckbox.isSelected()) {
            // cmd.append("-clip_n_m ");
            commandArgs.add("-clip_n_m");
        }
        // cmd.append("-minLength ").append(minLengthSpinner.getValue()).append(" ");
        commandArgs.add("-minLength");
        commandArgs.add(minLengthSpinner.getValue().toString());
        // cmd.append("-maxLength ").append(maxLengthSpinner.getValue()).append(" ");
        commandArgs.add("-maxLength");
        commandArgs.add(maxLengthSpinner.getValue().toString());
        // cmd.append("-min_pep_mz ").append(minPepMzSpinner.getValue()).append(" ");
        commandArgs.add("-min_pep_mz");
        commandArgs.add(minPepMzSpinner.getValue().toString());
        // cmd.append("-max_pep_mz ").append(maxPepMzSpinner.getValue()).append(" ");
        commandArgs.add("-max_pep_mz");
        commandArgs.add(maxPepMzSpinner.getValue().toString());
        // cmd.append("-min_pep_charge
        // ").append(minPepChargeSpinner.getValue()).append(" ");
        commandArgs.add("-min_pep_charge");
        commandArgs.add(minPepChargeSpinner.getValue().toString());
        // cmd.append("-max_pep_charge
        // ").append(maxPepChargeSpinner.getValue()).append(" ");
        commandArgs.add("-max_pep_charge");
        commandArgs.add(maxPepChargeSpinner.getValue().toString());
        commandArgs.add("-lf_frag_mz_min");
        commandArgs.add(libMinFragMzSpinner.getValue().toString());
        // cmd.append("-lf_frag_mz_max ").append(maxFragMzSpinner.getValue()).append("
        // ");
        commandArgs.add("-lf_frag_mz_max");
        commandArgs.add(libMaxFragMzSpinner.getValue().toString());
        // cmd.append("-lf_top_n_frag ").append(maxFragIonsSpinner.getValue()).append("
        // ");
        commandArgs.add("-lf_top_n_frag");
        commandArgs.add(LibTopNFragIonsSpinner.getValue().toString());

        // -lf_min_n_frag
        commandArgs.add("-lf_min_n_frag");
        commandArgs.add(libMinNumFragSpinner.getValue().toString());

        // -lf_frag_n_min
        commandArgs.add("-lf_frag_n_min");
        commandArgs.add(libFragNumMinSpinner.getValue().toString());

        // cmd.append("-lf_type ").append(libraryFormatCombo.getSelectedItem()).append("
        // ");
        commandArgs.add("-lf_type");
        commandArgs.add(libraryFormatCombo.getSelectedItem().toString());
        // cmd.append("-se DIA-NN ");

        commandArgs.add("-se");
        commandArgs.add("DIA-NN");

        if (!trainMsFile.isEmpty()) {
            // cmd.append("-tf all ");
            commandArgs.add("-tf");
            commandArgs.add("all");
        }

        if (additionalOptionList.contains("-nm")) {
            commandArgs.add("-nm");
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-nm"));
        } else if (additionalOptionList.contains("!-nm")) {
            //
            additionalOptionInGuiList.add(additionalOptionList.indexOf("!-nm"));
        } else {
            commandArgs.add("-nm");
        }

        if (additionalOptionList.contains("-nf")) {
            commandArgs.add("-nf");
            String nfValue = additionalOptionList.get(additionalOptionList.indexOf("-nf") + 1);
            try {
                Integer.parseInt(nfValue);
                commandArgs.add(nfValue);
            } catch (NumberFormatException nfe) {
                commandArgs.add("4");
            }
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-nf"));
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-nf") + 1);
        } else {
            commandArgs.add("-nf");
            commandArgs.add("4");
        }

        if (additionalOptionList.contains("-min_n")) {
            commandArgs.add("-min_n");
            String nfValue = additionalOptionList.get(additionalOptionList.indexOf("-min_n") + 1);
            try {
                Integer.parseInt(nfValue);
                commandArgs.add(nfValue);
            } catch (NumberFormatException nfe) {
                commandArgs.add("4");
            }
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-min_n"));
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-min_n") + 1);
        } else {
            commandArgs.add("-min_n");
            commandArgs.add("4");
        }

        if (additionalOptionList.contains("-valid")) {
            commandArgs.add("-valid");
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-valid"));
        } else if (additionalOptionList.contains("!-valid")) {
            //
            additionalOptionInGuiList.add(additionalOptionList.indexOf("!-valid"));
        } else {
            commandArgs.add("-valid");
        }

        if (additionalOptionList.contains("-na")) {
            commandArgs.add("-na");
            String naValue = additionalOptionList.get(additionalOptionList.indexOf("-na") + 1);
            try {
                Integer.parseInt(naValue);
                commandArgs.add(naValue);
            } catch (NumberFormatException nfe) {
                commandArgs.add("0");
            }
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-na"));
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-na") + 1);
        } else {
            commandArgs.add("-na");
            commandArgs.add("0");
        }

        if (additionalOptionList.contains("-ez")) {
            commandArgs.add("-ez");
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-ez"));
        } else if (additionalOptionList.contains("!-ez")) {
            //
            additionalOptionInGuiList.add(additionalOptionList.indexOf("!-ez"));
        } else {
            commandArgs.add("-ez");
        }

        if (additionalOptionList.contains("-fast")) {
            commandArgs.add("-fast");
            additionalOptionInGuiList.add(additionalOptionList.indexOf("-fast"));
        } else if (additionalOptionList.contains("!-fast")) {
            //
            additionalOptionInGuiList.add(additionalOptionList.indexOf("!-fast"));
        } else {
            commandArgs.add("-fast");
        }

        if (!additionalOptionList.isEmpty()) {
            // remove additional options that are already in diannArgs
            // Sort indexes in descending order
            additionalOptionInGuiList.sort(Collections.reverseOrder());
            for (int index : additionalOptionInGuiList) {
                if (index >= 0 && index < additionalOptionList.size()) {
                    additionalOptionList.remove(index);
                }
            }
            // check if any of remaining additional options are in diannArgs
            // start with --
            for (String option : additionalOptionList) {
                if (option.startsWith("-")) {
                    if (commandArgs.contains(option)) {
                        // show warning message
                        JOptionPane.showMessageDialog(this, "The additional Carafe option " + option + " is redundant!",
                                "Carafe setting", JOptionPane.ERROR_MESSAGE);
                        return null;
                    }
                }
                commandArgs.add(option);
                // if (option.contains(" ")) {
                // commandArgs.add("\"" + option + "\"");
                // } else {
                // commandArgs.add(option);
                // }
            }
        }

        CmdTask cmdTask = new CmdTask(commandArgs, "Carafe", "Run Carafe for fine-tuned spectral library generation");
        cmdTask.cmd = StringUtils.join(commandArgs, " ");

        return cmdTask;
    }

    private String getJavaExecutable() {
        try {
            Optional<String> cmd = java.lang.ProcessHandle.current().info().command();
            if (cmd.isPresent()) {
                return cmd.get();
            }
        } catch (Throwable ignored) {
        }
        String javaHome = System.getProperty("java.home");
        String sep = System.getProperty("file.separator");
        return javaHome + sep + "bin" + sep
                + (System.getProperty("os.name").toLowerCase().contains("win") ? "java.exe" : "java");
    }

    private void runCarafe() {
        if (isRunning) {
            JOptionPane.showMessageDialog(this, "A process is already running!", "Warning",
                    JOptionPane.WARNING_MESSAGE);
            return;
        }

        int workflow = workflowCombo.getSelectedIndex();

        if (!validateInputs(workflow)) {
            return;
        }

        setInputsFrozen(true);

        String outDir = outputDirField.getText().trim();
        // Validation handled by validateInputs

        // Initialize log writer
        try {
            // Ensure outDir exists first if not already
            File outDirFile = new File(outDir);
            if (!outDirFile.exists()) {
                outDirFile.mkdirs();
            }
            logWriter = new BufferedWriter(new FileWriter(outDir + File.separator + "carafe_log.txt"));
        } catch (IOException e) {
            logToConsole("Failed to create log file: " + e.getMessage() + "\n");
        }

        logToConsole("Workflow: " + (workflow + 1) + "\n");
        // export date and time
        String date = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
        logToConsole("Date: " + date + "\n");

        switch (workflow) {
            case 0 -> {
                String trainMsFile = trainMsFileField.getText().trim();
                String trainDb = trainDbFileField.getText().trim();
                // Validations handled by validateInputs

                String diann_train_dir = outDir + File.separator + "diann_train";

                File diannTrainDirFile = new File(diann_train_dir);
                if (!diannTrainDirFile.exists()) {
                    diannTrainDirFile.mkdirs();
                }

                // Initialize log writer
                try {
                    // Ensure outDir exists first if not already
                    File outDirFile = new File(outDir);
                    if (!outDirFile.exists()) {
                        outDirFile.mkdirs();
                    }
                    logWriter = new BufferedWriter(new FileWriter(outDir + File.separator + "carafe_log.txt"));
                } catch (IOException e) {
                    logToConsole("Failed to create log file: " + e.getMessage() + "\n");
                }

                // Check for RAW conversion logic
                CmdTask conversionTask = null;
                java.util.List<String> finalMsFiles = new ArrayList<>();
                java.util.List<String> rawFilesToConvert = new ArrayList<>();

                // Resolve inputs
                if (!trainMsFiles.isEmpty()) {
                    for (String path : trainMsFiles) {
                        if (path.toLowerCase().endsWith(".raw"))
                            rawFilesToConvert.add(path);
                        else
                            finalMsFiles.add(path);
                    }
                } else if (!trainMsFile.isEmpty()) {
                    File trainFileObj = new File(trainMsFile);
                    if (trainFileObj.isDirectory()) {
                        boolean hasMzML = false;
                        File[] mzMLs = trainFileObj.listFiles((dir, name) -> name.toLowerCase().endsWith(".mzml"));
                        if (mzMLs != null && mzMLs.length > 0) {
                            hasMzML = true;
                            for (File f : mzMLs)
                                finalMsFiles.add(f.getAbsolutePath());
                        }

                        if (!hasMzML) {
                            File[] rawFiles = trainFileObj
                                    .listFiles((dir, name) -> name.toLowerCase().endsWith(".raw"));
                            if (rawFiles != null) {
                                for (File f : rawFiles)
                                    rawFilesToConvert.add(f.getAbsolutePath());
                            }
                        }
                    } else if (trainMsFile.toLowerCase().endsWith(".raw")) {
                        rawFilesToConvert.add(trainMsFile);
                    } else {
                        finalMsFiles.add(trainMsFile);
                    }
                }

                // Setup Conversion Task if needed
                if (!rawFilesToConvert.isEmpty()) {
                    String subDir = outDir + File.separator + "train_mzML";
                    File subDirFile = new File(subDir);
                    if (!subDirFile.exists())
                        subDirFile.mkdirs();

                    String convCmd = buildMsConvertCommand(rawFilesToConvert, subDir);
                    conversionTask = new CmdTask(convCmd, "MSConvert", "Convert RAW files to mzML");

                    for (String rawPath : rawFilesToConvert) {
                        String rawName = new File(rawPath).getName();
                        String baseName = rawName.lastIndexOf('.') > 0
                                ? rawName.substring(0, rawName.lastIndexOf('.'))
                                : rawName;
                        finalMsFiles.add(subDir + File.separator + baseName + ".mzML");
                    }
                }

                if (finalMsFiles.isEmpty()) {
                    JOptionPane.showMessageDialog(this,
                            "Please select a valid mzML/timsTOF DIA file or a folder containing mzML files or timsTOF DIA raw files.",
                            "Input Required", JOptionPane.WARNING_MESSAGE);
                    return;
                }

                CmdTask diannTask = buildDIANNCommand(finalMsFiles, "", trainDb, diann_train_dir);
                if (diannTask != null) {
                    diannTask.task_description = "Run DIA-NN search on the training MS data";
                }

                String effectiveMsFile = "";
                if (!finalMsFiles.isEmpty()) {
                    if (finalMsFiles.size() == 1) {
                        effectiveMsFile = finalMsFiles.getFirst();
                    } else {
                        effectiveMsFile = new File(finalMsFiles.getFirst()).getParent();
                        // need to check if all files are in the same directory
                        for (String file : finalMsFiles) {
                            if (!new File(file).getParent().equals(effectiveMsFile)) {
                                effectiveMsFile = "";
                                break;
                            }
                        }
                        if (effectiveMsFile.isEmpty()) {
                            JOptionPane.showMessageDialog(this,
                                    "All files must be in the same directory.",
                                    "Input Required", JOptionPane.WARNING_MESSAGE);
                            return;
                        }
                    }
                }
                String diann_report_file;
                if (isDiannV2) {
                    diann_report_file = diann_train_dir + File.separator + "report.parquet";
                } else {
                    diann_report_file = diann_train_dir + File.separator + "report.tsv";
                }
                // System.out.println("DIANN report file: " + diann_report_file);

                if (tabbedPane != null) {
                    SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(4));
                }

                CmdTask[] initialTasks;
                if (conversionTask != null) {
                    initialTasks = new CmdTask[] { conversionTask, diannTask };
                } else {
                    initialTasks = new CmdTask[] { diannTask };
                }

                final String finalEffectiveMsFile = effectiveMsFile;
                executeChainedCommands(initialTasks, () -> {
                    final CmdTask[] commandContainer = new CmdTask[1];
                    try {
                        SwingUtilities.invokeAndWait(() -> {
                            diannReportFileField.setText(diann_report_file);
                            // System.out.println("DIANN report file: " + diann_report_file);
                            CmdTask carafe_task = buildCarafeCommand(finalEffectiveMsFile);
                            if (carafe_task != null) {
                                carafe_task.task_description = "Run Carafe to generate fine-tuned library";
                            }
                            commandContainer[0] = carafe_task;
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    return new CmdTask[] { commandContainer[0] };
                });
            }
            case 1 -> {
                // Check for RAW conversion logic even for Workflow 1 if trainMsFile is
                // populated and RAW
                // Resolve trainMsFile: check list first, then text field
                String trainMsFile;
                if (!trainMsFiles.isEmpty()) {
                    // Multi-file selection: use the parent folder of the first file
                    if (trainMsFiles.size() >= 2) {
                        trainMsFile = new File(trainMsFiles.getFirst()).getParent();
                    } else {
                        trainMsFile = trainMsFiles.getFirst();
                    }
                } else {
                    trainMsFile = trainMsFileField.getText().trim();
                }

                // Check for RAW files - not supported for Workflow 2
                boolean hasRawFiles = false;
                if (trainMsFile.toLowerCase().endsWith(".raw")) {
                    hasRawFiles = true;
                } else if (new File(trainMsFile).isDirectory()) {
                    File trainFileObj = new File(trainMsFile);
                    File[] rawFiles = trainFileObj.listFiles((dir, name) -> name.toLowerCase().endsWith(".raw"));
                    File[] mzMLFiles = trainFileObj.listFiles((dir, name) -> name.toLowerCase().endsWith(".mzml"));
                    if (rawFiles != null && rawFiles.length > 0 && (mzMLFiles == null || mzMLFiles.length == 0)) {
                        hasRawFiles = true;
                    }
                }

                if (hasRawFiles) {
                    JOptionPane.showMessageDialog(this,
                            "RAW files are not supported for Workflow 2.\n" +
                                    "The train MS file(s) must be mzML format, and\n" +
                                    "the DIA-NN report file must be generated using the same mzML files(s).",
                            "RAW Files Not Supported", JOptionPane.WARNING_MESSAGE);
                    setInputsFrozen(false);
                    return;
                }

                CmdTask carafeTask = buildCarafeCommand(null);
                if (carafeTask != null) {
                    carafeTask.task_description = "Run Carafe to generate spectral library";
                }
                executeCommand(carafeTask);
            }
            case 2 -> {
                // Collect Train MS Files
                java.util.List<String> effectiveTrainFiles = new java.util.ArrayList<>();
                if (!trainMsFiles.isEmpty()) {
                    effectiveTrainFiles.addAll(trainMsFiles);
                } else if (!trainMsFileField.getText().trim().isEmpty()) {
                    effectiveTrainFiles.add(trainMsFileField.getText().trim());
                }

                if (effectiveTrainFiles.isEmpty()) {
                    // Handled by validateInputs
                }

                // Collect Project MS Files
                java.util.List<String> effectiveProjectFiles = new java.util.ArrayList<>();
                if (!projectMsFiles.isEmpty()) {
                    effectiveProjectFiles.addAll(projectMsFiles);
                } else if (!projectMsFileField.getText().trim().isEmpty()) {
                    String projectPath = projectMsFileField.getText().trim();
                    File projectFile = new File(projectPath);
                    if (projectFile.isDirectory()) {
                        // Expand folder to individual files (RAW or mzML)
                        // check if there are mzML files in the folder first
                        File[] mzMLFiles = projectFile.listFiles((dir, name) -> name.toLowerCase().endsWith(".mzml"));
                        if (mzMLFiles != null && mzMLFiles.length > 0) {
                            for (File f : mzMLFiles) {
                                effectiveProjectFiles.add(f.getAbsolutePath());
                            }
                        } else {
                            // check if there are raw files in the folder
                            File[] rawFiles = projectFile.listFiles((dir, name) -> name.toLowerCase().endsWith(".raw"));
                            if (rawFiles != null) {
                                for (File f : rawFiles) {
                                    effectiveProjectFiles.add(f.getAbsolutePath());
                                }
                            }
                        }
                    } else {
                        effectiveProjectFiles.add(projectPath);
                    }
                }

                if (effectiveProjectFiles.isEmpty()) {
                    JOptionPane.showMessageDialog(this,
                            "No project MS files found.",
                            "No Project MS Files", JOptionPane.WARNING_MESSAGE);
                    setInputsFrozen(false);
                    return;
                }

                String trainDb = trainDbFileField.getText().trim();
                String libraryDb = libraryDbFileField.getText().trim();

                String diann_train_dir = outDir + File.separator + "diann_train";
                File diannTrainDirFile = new File(diann_train_dir);
                if (!diannTrainDirFile.exists()) {
                    diannTrainDirFile.mkdirs();
                }

                // Check for RAW conversion logic
                CmdTask conversionTask = null;
                List<String> finalTrainMzMLFiles = new ArrayList<>();
                String singleEffectiveMsFile = null; // Used only if we have a single file/folder path string logic for
                                                     // compatibility

                // Logic:
                // 1. If effectiveTrainFiles contains files (from List or single file from
                // text), process them.
                // 2. If it is a single entry that is a DIRECTORY, fall back to directory
                // scanning logic.

                boolean isDirectory = false;
                if (effectiveTrainFiles.size() == 1) {
                    if (new File(effectiveTrainFiles.get(0)).isDirectory()) {
                        isDirectory = true;
                    }
                }

                if (isDirectory) {
                    // Fallback to existing directory scanning logic
                    File trainFileObj = new File(effectiveTrainFiles.get(0));
                    boolean hasMzML = false;
                    File[] mzMLs = trainFileObj.listFiles((dir, name) -> name.toLowerCase().endsWith(".mzml"));
                    if (mzMLs != null && mzMLs.length > 0)
                        hasMzML = true;

                    if (!hasMzML) {
                        File[] rawFiles = trainFileObj.listFiles((dir, name) -> name.toLowerCase().endsWith(".raw"));
                        if (rawFiles != null && rawFiles.length > 0) {
                            String subDir = outDir + File.separator + "train_mzML";
                            File subDirFile = new File(subDir);
                            if (!subDirFile.exists())
                                subDirFile.mkdirs();

                            String wildcardPath = trainFileObj.getAbsolutePath() + File.separator + "*.raw";
                            String convCmd = buildMsConvertCommand(wildcardPath, subDir);
                            conversionTask = new CmdTask(convCmd, "MSConvert",
                                    "Convert RAW files in directory to mzML");

                            for (File raw : rawFiles) {
                                String rawName = raw.getName();
                                String baseName = rawName.lastIndexOf('.') > 0
                                        ? rawName.substring(0, rawName.lastIndexOf('.'))
                                        : rawName;
                                finalTrainMzMLFiles.add(subDir + File.separator + baseName + ".mzML");
                            }
                            singleEffectiveMsFile = subDir; // Pass subdir for searching
                        }
                    } else {
                        singleEffectiveMsFile = trainFileObj.getAbsolutePath();
                    }
                } else {
                    // File List Logic (Single or Multiple)
                    boolean isRaw = effectiveTrainFiles.stream().anyMatch(f -> f.toLowerCase().endsWith(".raw"));

                    if (isRaw) {
                        // Convert all RAW files
                        String subDir = outDir + File.separator + "train_mzML";
                        File subDirFile = new File(subDir);
                        if (!subDirFile.exists())
                            subDirFile.mkdirs();

                        String convCmd = buildMsConvertCommand(effectiveTrainFiles, subDir);
                        conversionTask = new CmdTask(convCmd, "MSConvert", "Convert RAW files to mzML");

                        for (String f : effectiveTrainFiles) {
                            String rawName = new File(f).getName();
                            String baseName = rawName.contains(".") ? rawName.substring(0, rawName.lastIndexOf('.'))
                                    : rawName;
                            finalTrainMzMLFiles.add(subDir + File.separator + baseName + ".mzML");
                        }
                    } else {
                        finalTrainMzMLFiles.addAll(effectiveTrainFiles);
                    }
                }

                CmdTask diannTask;
                if (!finalTrainMzMLFiles.isEmpty()) {
                    diannTask = buildDIANNCommand(finalTrainMzMLFiles, "", trainDb, diann_train_dir);
                } else {
                    diannTask = buildDIANNCommand(singleEffectiveMsFile, "", trainDb, diann_train_dir,
                            conversionTask != null);
                }

                if (diannTask != null) {
                    diannTask.task_description = "Run DIA-NN search on the training MS data";
                }
                String diann_report_file;
                if (this.isDiannV2) {
                    diann_report_file = diann_train_dir + File.separator + "report.parquet";
                } else {
                    diann_report_file = diann_train_dir + File.separator + "report.tsv";
                }

                String diann_project_dir = outDir + File.separator + "diann_project";
                File diannProjectDirFile = new File(diann_project_dir);
                if (!diannProjectDirFile.exists()) {
                    diannProjectDirFile.mkdirs();
                }
                final String carafeLibraryPath = outDir + File.separator + "carafe_library" + File.separator
                        + "SkylineAI_spectral_library.tsv";

                if (tabbedPane != null) {
                    SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(4));
                }

                CmdTask[] initialTasks;
                if (conversionTask != null) {
                    initialTasks = new CmdTask[] { conversionTask, diannTask };
                } else {
                    initialTasks = new CmdTask[] { diannTask };
                }

                String carafeMsInput = null;
                if (!finalTrainMzMLFiles.isEmpty()) {
                    if (finalTrainMzMLFiles.size() == 1) {
                        carafeMsInput = finalTrainMzMLFiles.getFirst();
                    } else {
                        File first = new File(finalTrainMzMLFiles.getFirst());
                        carafeMsInput = first.getParent();
                        // If we converted, they are in subDir, so no need to check.
                        if (conversionTask == null) {
                            // need to check if all files are in the same directory
                            for (String file : finalTrainMzMLFiles) {
                                if (!new File(file).getParent().equals(carafeMsInput)) {
                                    carafeMsInput = "";
                                    break;
                                }
                            }
                            if (carafeMsInput.isEmpty()) {
                                JOptionPane.showMessageDialog(this,
                                        "All files must be in the same directory.",
                                        "Input Required", JOptionPane.WARNING_MESSAGE);
                                return;
                            }
                        }
                    }
                } else {
                    carafeMsInput = singleEffectiveMsFile;
                }

                final String finalCarafeMsInput = carafeMsInput;

                executeChainedCommands(initialTasks, () -> {
                    final CmdTask[] commands = new CmdTask[2];
                    try {
                        SwingUtilities.invokeAndWait(() -> {
                            diannReportFileField.setText(diann_report_file);
                            CmdTask carafe_task = buildCarafeCommand(finalCarafeMsInput);
                            if (carafe_task != null) {
                                carafe_task.task_description = "Run Carafe to generate fine-tuned library";
                            }
                            commands[0] = carafe_task;

                            // For the project search step, we also need to handle multiple project files!
                            // New buildDIANNCommand supports List<String>
                            if (!effectiveProjectFiles.isEmpty()) {
                                commands[1] = buildDIANNCommand(effectiveProjectFiles, carafeLibraryPath, libraryDb,
                                        diann_project_dir);
                            } else {
                                // Fallback?? Should verify handled earlier.
                                commands[1] = null;
                            }

                            if (commands[1] != null) {
                                commands[1].task_description = "DIA-NN search for project data using fine-tuned library";
                            }
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                        return new CmdTask[0];
                    }
                    return commands;
                });
            }
            default -> JOptionPane.showMessageDialog(this, "Unsupported workflow selected!", "Error",
                    JOptionPane.ERROR_MESSAGE);
        }
    }

    @FunctionalInterface
    private interface NextCommandsSupplier {
        CmdTask[] getNextCommands();
    }

    private void executeChainedCommands(CmdTask[] initialCommands, NextCommandsSupplier nextCommandsSupplier) {
        isRunning = true;
        runButton.setEnabled(false);
        stopButton.setEnabled(true);
        progressBar.setIndeterminate(true);
        progressBar.setString("Running DIA-NN...");

        timeUsageMap.clear();

        Object selectedPython = pythonPathCombo.getSelectedItem();
        String pythonPath = selectedPython != null ? selectedPython.toString().trim() : "";
        if (!pythonPath.isEmpty()) {
            prefs.put(PREF_PYTHON_PATH, pythonPath);
        }

        // Automatically save screenshots of parameter panels at the start
        // Ensure this runs on EDT before background thread starts
        saveParameterScreenshots();

        executor = Executors.newSingleThreadExecutor();
        executor.submit(() -> {
            try {
                int stepIndex = 1;
                for (CmdTask command : initialCommands) {
                    if (!isRunning)
                        return;

                    updateProgressBarForCommand(command.task_description);

                    logToConsole("\n========================================\n");
                    logToConsole("Running: " + command.task_description + "\n");
                    logToConsole("Command: " + command.cmd + "\n");
                    logToConsole("========================================\n\n");

                    long start = System.nanoTime();
                    int exitCode = runSingleCommand(command, pythonPath);
                    long end = System.nanoTime();
                    double minutes = (end - start) / 1e9 / 60.0;
                    String key = String.format("%02d. %s - %s", stepIndex++, command.task_name,
                            command.task_description);
                    timeUsageMap.put(key, minutes);

                    if (exitCode != 0) {
                        SwingUtilities.invokeLater(() -> {
                            logToConsole("\n[ERROR] Command failed with exit code: " + exitCode + "\n");
                            progressBar.setString("Failed");
                            finishExecution();
                        });
                        return;
                    }
                }

                if (nextCommandsSupplier != null && isRunning) {
                    CmdTask[] nextCommands = nextCommandsSupplier.getNextCommands();
                    for (CmdTask command : nextCommands) {
                        if (!isRunning)
                            return;

                        updateProgressBarForCommand(command.task_description);

                        logToConsole("\n========================================\n");
                        logToConsole("Running: " + command.task_description + "\n");
                        logToConsole("Command: " + command.cmd + "\n");
                        logToConsole("========================================\n\n");
                        long start = System.nanoTime();
                        int exitCode = runSingleCommand(command, pythonPath);
                        long end = System.nanoTime();
                        double minutes = (end - start) / 1e9 / 60.0;
                        String key = String.format("%02d. %s - %s", stepIndex++, command.task_name,
                                command.task_description);
                        timeUsageMap.put(key, minutes);
                        if (exitCode != 0) {
                            SwingUtilities.invokeLater(() -> {
                                logToConsole("\n[ERROR] Command failed with exit code: " + exitCode + "\n");
                                progressBar.setString("Failed");
                                finishExecution();
                            });
                            return;
                        }
                    }
                }

                SwingUtilities.invokeLater(() -> {
                    logToConsole("\n[SUCCESS] Workflow completed successfully!\n");
                    progressBar.setString("Completed");
                    logToConsole("\n[SUMMARY] Step durations (min):\n");
                    double totalTime = 0.0;
                    for (java.util.Map.Entry<String, Double> e : timeUsageMap.entrySet()) {
                        logToConsole(" - " + e.getKey() + " : " + String.format("%.2f", e.getValue()) + "\n");
                        totalTime += e.getValue();
                    }
                    logToConsole("Total time: " + String.format("%.2f", totalTime) + " min\n");
                    finishExecution();
                });

            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    logToConsole("\n[ERROR] Error: " + e.getMessage() + "\n");
                    progressBar.setString("Error");
                    finishExecution();
                });
            }
        });
    }

    private void updateProgressBarForCommand(String description) {
        if (progressBar == null)
            return;
        SwingUtilities.invokeLater(() -> progressBar.setString(description));
    }

    private int runSingleCommand(CmdTask task, String pythonPath) throws Exception {
        ProcessBuilder pb = new ProcessBuilder();
        String commandString = (task.cmd != null) ? task.cmd : String.join(" ", task.args);

        if (task.args != null && !task.args.isEmpty()) {
            pb.command(task.args);
        } else if (System.getProperty("os.name").toLowerCase().contains("windows")) {
            // Wrap the entire command in quotes to handle potential spaces/quotes issues
            // with cmd /C
            pb.command("cmd", "/c", "\"" + task.cmd + "\"");
        } else {
            pb.command("bash", "-c", task.cmd);
        }
        pb.redirectErrorStream(true);

        if (!pythonPath.isEmpty()) {
            java.util.Map<String, String> env = pb.environment();
            File pythonFile = new File(pythonPath);
            String pythonDir = pythonFile.isFile() ? pythonFile.getParent() : pythonPath;
            if (pythonDir != null) {
                String pathSeparator = System.getProperty("os.name").toLowerCase().contains("windows") ? ";" : ":";
                String currentPath = env.getOrDefault("PATH", env.getOrDefault("Path", ""));
                String newPath = pythonDir + pathSeparator + currentPath;
                env.put("PATH", newPath);
                if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                    env.put("Path", newPath);
                }
            }
        }

        String lowerCmd = commandString.toLowerCase();
        // check if this is a diann command
        boolean is_a_diann_task = false;
        if (lowerCmd.contains("diann") && lowerCmd.contains("--f ")) {
            is_a_diann_task = true;
            java.util.Map<String, String> env = pb.environment();
            String target_omp_num_threads = "OMP_NUM_THREADS";
            String target_mkl_num_threads = "MKL_NUM_THREADS";
            String target_kmp_affinity = "KMP_AFFINITY";
            try {
                for (String key : env.keySet()) {
                    if (key.equalsIgnoreCase(target_omp_num_threads)) {
                        target_omp_num_threads = key;
                        break;
                    }
                }
                for (String key : env.keySet()) {
                    if (key.equalsIgnoreCase(target_mkl_num_threads)) {
                        target_mkl_num_threads = key;
                        break;
                    }
                }
                for (String key : env.keySet()) {
                    if (key.equalsIgnoreCase(target_kmp_affinity)) {
                        target_kmp_affinity = key;
                        break;
                    }
                }

                String omp = env.get(target_omp_num_threads);
                if (omp == null || omp.trim().isEmpty() || omp.trim().equals("0")) {
                    String threads = String.valueOf(Runtime.getRuntime().availableProcessors());
                    env.put(target_omp_num_threads, threads);
                    env.put(target_mkl_num_threads, threads);
                }
                env.remove(target_kmp_affinity);
                env.put("KMP_WARNINGS", "off");
            } catch (Throwable ignored) {
            }

            String dbgOmp = pb.environment().getOrDefault(target_omp_num_threads, "(unset)");
            String dbgMkl = pb.environment().getOrDefault(target_mkl_num_threads, "(unset)");
            String dbgKmp = pb.environment().getOrDefault(target_kmp_affinity, "(unset)");
            final String dbgMsg = String.format(
                    "[DEBUG] DIANN env: OMP_NUM_THREADS=%s, MKL_NUM_THREADS=%s, KMP_AFFINITY=%s", dbgOmp, dbgMkl,
                    dbgKmp);
            logToConsole(dbgMsg + "\n");
        }

        currentProcess = pb.start();

        boolean errorDetected = false;
        BufferedReader reader = new BufferedReader(new InputStreamReader(currentProcess.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            if (is_a_diann_task) {
                // currently only perform this check for DIANN tasks
                String trimmed = line.trim();
                if (trimmed.startsWith("ERROR:") || trimmed.startsWith("Error:")) {
                    errorDetected = true;
                }
            }
            final String output = line;
            logToConsole(output + "\n");
        }

        int exitCode = currentProcess.waitFor();
        if (exitCode == 0 && errorDetected) {
            return -1;
        }
        return exitCode;
    }

    private CmdTask buildDIANNCommand(String ms_file, String spectral_library_file, String database, String out_dir,
            boolean bypassFileCheck) {
        List<String> msFiles = new ArrayList<>();
        if (bypassFileCheck) {
            msFiles.add(ms_file);
        } else {
            File F = new File(ms_file);
            if (F.isFile()) {
                msFiles.add(ms_file);
            } else if (F.isDirectory()) {
                File analysisTdf = new File(ms_file + File.separator + "analysis.tdf");
                if (analysisTdf.exists()) {
                    msFiles.add(ms_file);
                } else {
                    File[] mzMLFiles = F.listFiles(
                            (dir, name) -> name.toLowerCase().endsWith(".mzml") || name.toLowerCase().endsWith(".raw"));
                    if (mzMLFiles != null && mzMLFiles.length > 0) {
                        for (File f : mzMLFiles)
                            msFiles.add(f.getAbsolutePath());
                    } else {
                        File[] subDirs = F.listFiles(File::isDirectory);
                        if (subDirs != null) {
                            for (File subDir : subDirs) {
                                File subAnalysisTdf = new File(subDir.getPath() + File.separator + "analysis.tdf");
                                if (subAnalysisTdf.exists()) {
                                    msFiles.add(subDir.getAbsolutePath());
                                }
                            }
                        }
                    }
                }
            }
        }

        if (msFiles.isEmpty()) {
            JOptionPane.showMessageDialog(this,
                    "Please select a valid mzML/timsTOF DIA file or a folder containing mzML files or timsTOF DIA raw files.",
                    "Input Required", JOptionPane.WARNING_MESSAGE);
            return null;
        }

        return buildDIANNCommand(msFiles, spectral_library_file, database, out_dir);
    }

    private CmdTask buildDIANNCommand(java.util.List<String> msFiles, String spectral_library_file,
            String database,
            String out_dir) {
        Object diannPath = diannPathCombo.getSelectedItem();
        ArrayList<String> diannArgs = new ArrayList<>();
        if (diannPath != null && !diannPath.toString().trim().isEmpty()) {

            // DIA-NN additional arguments
            String additionalOptions = diannAdditionalOptionsField.getText().trim();
            ArrayList<String> additionalOptionList = new ArrayList<>();
            // store the index of the additional options which are present through the GUI
            ArrayList<Integer> additionalOptionInGuiList = new ArrayList<>();
            if (!additionalOptions.isEmpty()) {
                String[] additional_options = Commandline.translateCommandline(additionalOptions);
                Collections.addAll(additionalOptionList, additional_options);
            }

            // String diann_path = "\"" + diannPath.toString().trim() + "\"";
            String diann_path = diannPath.toString().trim();
            diannArgs.add(diann_path);

            String version = getDIANNVersion(diann_path);
            diannVersion = version;
            boolean isV2 = true;
            try {
                String[] vParts = version.split("\\.");
                if (vParts.length > 0) {
                    int major = Integer.parseInt(vParts[0]);
                    if (major < 2) {
                        isV2 = false;
                    }
                }
            } catch (Exception e) {
                // ignore
            }
            isDiannV2 = isV2;

            for (String f : msFiles) {
                diannArgs.add("--f");
                diannArgs.add(f);
                // diannArgs.add("\"" + f + "\"");
            }

            if (spectral_library_file.isEmpty() && !database.isEmpty()) {
                diannArgs.add("--lib");
                diannArgs.add("");
                // diannArgs.add("\"\"");
                diannArgs.add("--gen-spec-lib");
                diannArgs.add("--predictor");
                diannArgs.add("--fasta");
                diannArgs.add(database);
                // diannArgs.add("\"" + database + "\"");
                diannArgs.add("--fasta-search");
            } else if (!spectral_library_file.isEmpty() && !database.isEmpty()) {
                diannArgs.add("--lib");
                diannArgs.add(spectral_library_file);
                // diannArgs.add("\"" + spectral_library_file + "\"");
                diannArgs.add("--gen-spec-lib");
                diannArgs.add("--reannotate");
                diannArgs.add("--fasta");
                diannArgs.add(database);
                // diannArgs.add("\"" + database + "\"");
            } else {
                JOptionPane.showMessageDialog(this,
                        "Please provide a spectral library file or a protein database file.", "Input Required",
                        JOptionPane.WARNING_MESSAGE);
                return null;
            }

            int cores = Runtime.getRuntime().availableProcessors();
            diannArgs.add("--threads");
            diannArgs.add(String.valueOf(cores));
            diannArgs.add("--verbose");
            if (additionalOptionList.contains("--verbose")) {
                int index = additionalOptionList.indexOf("--verbose");
                if (index < additionalOptionList.size() - 1) {
                    String verboseValue = additionalOptionList.get(index + 1);
                    // check if this is a number
                    try {
                        Integer.parseInt(verboseValue);
                        diannArgs.add(verboseValue);
                    } catch (NumberFormatException nfe) {
                        diannArgs.add("1");
                    }
                    additionalOptionInGuiList.add(index);
                    additionalOptionInGuiList.add(index + 1);
                } else {
                    diannArgs.add("1");
                    additionalOptionInGuiList.add(index);
                }
            } else {
                diannArgs.add("1");
            }

            diannArgs.add("--out");
            String ext = isV2 ? ".parquet" : ".tsv";
            diannArgs.add(out_dir + File.separator + "report" + ext);
            // diannArgs.add("\"" + out_dir + File.separator + "report.parquet\"");
            diannArgs.add("--out-lib");
            diannArgs.add(out_dir + File.separator + "report-lib" + ext);
            // diannArgs.add("\"" + out_dir + File.separator + "report-lib.parquet\"");

            String fixModSelected = fixModSelectedField.getText().trim();
            if (fixModSelected.equalsIgnoreCase("1")) {
                diannArgs.add("--unimod4");
            } else {
                JOptionPane.showMessageDialog(this,
                        "Unsupported modification settings. Please select '1' for Fixed modifications.", "Warning",
                        JOptionPane.WARNING_MESSAGE);
                return null;
            }

            String varModSelected = varModSelectedField.getText().trim();
            if (varModSelected.equalsIgnoreCase("2")) {
                diannArgs.add("--var-mods");
                if ((int) maxVarSpinner.getValue() >= 1) {
                    diannArgs.add(String.valueOf(maxVarSpinner.getValue()));
                } else {
                    // show warning message
                    JOptionPane.showMessageDialog(this,
                            "Please set maximum number of variable modifications to at least 1 when variable modification is set.",
                            "Warning", JOptionPane.WARNING_MESSAGE);
                    return null;
                }

                diannArgs.add("--var-mod");
                diannArgs.add("UniMod:35,15.994915,M");
            } else if (varModSelected.equalsIgnoreCase("0")) {
                // no modification
            } else if(varModSelected.equalsIgnoreCase("7,8,9")){
                // 1.8.1: diann.exe --lib "" --threads 8 --verbose 1 --out "C:\tools\DIA-NN\1.8.1\report.tsv"        --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --monitor-mod UniMod:21 --reanalyse --relaxed-prot-inf --smart-profiling --peak-center --no-ifs-removal
                // 1.9.1: diann.exe --lib "" --threads 8 --verbose 1 --out "D:\software\DIA-NN\1.9.1/report.tsv"     --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --relaxed-prot-inf --rt-profiling
                // 1.9.2: diann.exe --lib "" --threads 8 --verbose 1 --out "D:\software\DIA-NN\1.9.2/report.tsv"     --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --relaxed-prot-inf --rt-profiling
                // 2.0:   diann.exe --lib "" --threads 8 --verbose 1 --out "D:\software\DIA-NN\2.0\report.parquet"   --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse --rt-profiling
                // 2.0.2: diann.exe --lib "" --threads 8 --verbose 1 --out "D:\software\DIA-NN\2.0.2\report.parquet" --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse --rt-profiling
                // 2.1.0: diann.exe --lib "" --threads 8 --verbose 1 --out "D:\software\DIA-NN\2.1.0\report.parquet" --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse --rt-profiling
                // 2.2.0: diann.exe --lib "" --threads 8 --verbose 1 --out "D:\software\DIA-NN\2.2.0\report.parquet" --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse --rt-profiling
                // 2.2.1: diann.exe --lib "" --threads 8 --verbose 1 --out "D:\software\DIA-NN\2.2.1\report.parquet" --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse --rt-profiling
                // 2.3.0: diann.exe --lib "" --threads 8 --verbose 1 --out "D:\software\DIA-NN\2.3.0\report.parquet" --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse --rt-profiling
                // 2.3.1: diann.exe --lib "" --threads 8 --verbose 1 --out "D:\software\DIA-NN\2.3.1\report.parquet" --qvalue 0.01 --matrices  --unimod4 --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse --rt-profiling
                // --var-mod UniMod:21,79.966331,STY --peptidoforms
                diannArgs.add("--var-mods");
                if ((int) maxVarSpinner.getValue() >= 1) {
                    diannArgs.add(String.valueOf(maxVarSpinner.getValue()));
                } else {
                    // show warning message
                    JOptionPane.showMessageDialog(this,
                            "Please set maximum number of variable modifications to at least 1 when variable modification is set.",
                            "Warning", JOptionPane.WARNING_MESSAGE);
                    return null;
                }
                diannArgs.add("--var-mod");
                diannArgs.add("UniMod:21,79.966331,STY");
                if (this.diannVersion.equalsIgnoreCase("1.8.1")) {
                    // --monitor-mod UniMod:21
                    diannArgs.add("--monitor-mod");
                    diannArgs.add("UniMod:21");
                } else {
                    diannArgs.add("--peptidoforms");
                }
            } else if (varModSelected.equalsIgnoreCase("2,7,8,9")) {
                diannArgs.add("--var-mods");
                if ((int) maxVarSpinner.getValue() >= 1) {
                    diannArgs.add(String.valueOf(maxVarSpinner.getValue()));
                } else {
                    // show warning message
                    JOptionPane.showMessageDialog(this,
                            "Please set maximum number of variable modifications to at least 1 when variable modification is set.",
                            "Warning", JOptionPane.WARNING_MESSAGE);
                    return null;
                }
                diannArgs.add("--var-mod");
                diannArgs.add("UniMod:21,79.966331,STY");
                diannArgs.add("--var-mod");
                diannArgs.add("UniMod:35,15.994915,M");
                if (this.diannVersion.equalsIgnoreCase("1.8.1")) {
                    // --monitor-mod UniMod:21
                    diannArgs.add("--monitor-mod");
                    diannArgs.add("UniMod:21");
                } else {
                    diannArgs.add("--peptidoforms");
                }
            } else if (varModSelected.equalsIgnoreCase("10")) {
                // --var-mod UniMod:121,114.042927,K --no-cut-after-mod UniMod:121
                // --peptidoforms
                diannArgs.add("--var-mods");
                if ((int) maxVarSpinner.getValue() >= 1) {
                    diannArgs.add(String.valueOf(maxVarSpinner.getValue()));
                } else {
                    // show warning message
                    JOptionPane.showMessageDialog(this,
                            "Please set maximum number of variable modifications to at least 1 when variable modification is set.",
                            "Warning", JOptionPane.WARNING_MESSAGE);
                    return null;
                }
                diannArgs.add("--var-mod");
                diannArgs.add("UniMod:121,114.042927,K");
                diannArgs.add("--no-cut-after-mod");
                diannArgs.add("UniMod:121");
                if (this.diannVersion.equalsIgnoreCase("1.8.1")) {
                    // --monitor-mod UniMod:21
                    diannArgs.add("--monitor-mod");
                    diannArgs.add("UniMod:121");
                } else {
                    diannArgs.add("--peptidoforms");
                }
            } else {
                JOptionPane.showMessageDialog(this,
                        "Unsupported modification settings. Please select '2' for Variable modifications.", "Warning",
                        JOptionPane.WARNING_MESSAGE);
                return null;
            }

            String enzyme = ((String) enzymeCombo.getSelectedItem()).split(":")[0];
            if (enzyme.equals("1")) {
                diannArgs.add("--cut");
                diannArgs.add("K*,R*");
            } else if (enzyme.equals("2")) {
                diannArgs.add("--cut");
                diannArgs.add("K*,R*");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else {
                JOptionPane.showMessageDialog(this,
                        "Unsupported enzyme settings. Please select '1' for trypsin or '2' for chymotrypsin.",
                        "Warning", JOptionPane.WARNING_MESSAGE);
                return null;
            }

            if (clipNmCheckbox.isSelected()) {
                diannArgs.add("--met-excision");
            }

            diannArgs.add("--min-pep-len");
            diannArgs.add(String.valueOf(minLengthSpinner.getValue()));
            diannArgs.add("--max-pep-len");
            diannArgs.add(String.valueOf(maxLengthSpinner.getValue()));
            diannArgs.add("--min-pr-mz");
            diannArgs.add(String.valueOf(minPepMzSpinner.getValue()));
            diannArgs.add("--max-pr-mz");
            diannArgs.add(String.valueOf(maxPepMzSpinner.getValue()));
            diannArgs.add("--min-pr-charge");
            diannArgs.add(String.valueOf(minPepChargeSpinner.getValue()));
            diannArgs.add("--max-pr-charge");
            diannArgs.add(String.valueOf(maxPepChargeSpinner.getValue()));
            diannArgs.add("--min-fr-mz");
            diannArgs.add(String.valueOf(libMinFragMzSpinner.getValue()));
            diannArgs.add("--max-fr-mz");
            diannArgs.add(String.valueOf(libMaxFragMzSpinner.getValue()));

            diannArgs.add("--qvalue");
            // check if --qvalue is present in the additional options
            if (additionalOptionList.contains("--qvalue")) {
                int index = additionalOptionList.indexOf("--qvalue");
                if (index < additionalOptionList.size() - 1) {
                    String qvalueValue = additionalOptionList.get(index + 1);
                    // check if this is a number
                    try {
                        Double.parseDouble(qvalueValue);
                        diannArgs.add(qvalueValue);
                    } catch (NumberFormatException nfe) {
                        diannArgs.add("0.01");
                    }
                    additionalOptionInGuiList.add(index);
                    additionalOptionInGuiList.add(index + 1);
                } else {
                    diannArgs.add("0.01");
                    additionalOptionInGuiList.add(index);
                }
            } else {
                diannArgs.add("0.01");
            }
            diannArgs.add("--matrices");

            if (msFiles.size() >= 2) {
                diannArgs.add("--reanalyse");
            }

            // check if --smart-profiling is present in the additional options
            if (additionalOptionList.contains("--smart-profiling")) {
                diannArgs.add("--smart-profiling");
                additionalOptionInGuiList.add(additionalOptionList.indexOf("--smart-profiling"));
            } else if (additionalOptionList.contains("--id-profiling")) {
                // --id-profiling is present in the additional options
                diannArgs.add("--id-profiling");
                additionalOptionInGuiList.add(additionalOptionList.indexOf("--id-profiling"));
            } else if (additionalOptionList.contains("--rt-profiling")) {
                diannArgs.add("--rt-profiling");
                additionalOptionInGuiList.add(additionalOptionList.indexOf("--rt-profiling"));
            } else if (additionalOptionList.contains("!--rt-profiling")) {
                // full profiling
                additionalOptionInGuiList.add(additionalOptionList.indexOf("!--rt-profiling"));
            } else {
                diannArgs.add("--rt-profiling");
            }

            if (isV2) {
                diannArgs.add("--export-quant");
            }

            if (!additionalOptionList.isEmpty()) {
                // remove additional options that are already in diannArgs
                // Sort indexes in descending order
                additionalOptionInGuiList.sort(Collections.reverseOrder());
                for (int index : additionalOptionInGuiList) {
                    if (index >= 0 && index < additionalOptionList.size()) {
                        additionalOptionList.remove(index);
                    }
                }
                // check if any of remaining additional options are in diannArgs
                // start with --
                for (String option : additionalOptionList) {
                    if (option.startsWith("--")) {
                        if (diannArgs.contains(option)) {
                            // show warning message
                            JOptionPane.showMessageDialog(this,
                                    "The additional DIA-NN option " + option + " is redundant!", "DIA-NN setting",
                                    JOptionPane.ERROR_MESSAGE);
                            return null;
                        }
                    }
                    diannArgs.add(option);
                    // if (option.contains(" ")) {
                    // diannArgs.add("\"" + option + "\"");
                    // } else {
                    // diannArgs.add(option);
                    // }
                }
            }
            CmdTask task = new CmdTask(diannArgs, "DIA-NN", "Running DIA-NN");
            task.cmd = String.join(" ", diannArgs);
            return task;
        } else {
            JOptionPane.showMessageDialog(this, "Please provide a valid DIA-NN executable path.", "Input Required",
                    JOptionPane.WARNING_MESSAGE);
            return null;
        }
    }

    private String getDIANNVersion(String diannPath) {
        try {
            ProcessBuilder pb = new ProcessBuilder();
            if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                // Wrap in quotes for Windows cmd
                pb.command("cmd", "/c", "\"" + diannPath + "\"");
            } else {
                pb.command("bash", "-c", diannPath);
            }
            pb.redirectErrorStream(true);
            Process p = pb.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            String versionLine = null;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    versionLine = line.trim();
                    break;
                }
            }
            p.waitFor();
            // Expected format: "DIA-NN 2.3.1 Academia ..." or "DIA-NN 1.8.1 ..."
            if (versionLine != null && versionLine.startsWith("DIA-NN")) {
                String[] parts = versionLine.split("\\s+");
                if (parts.length >= 2) {
                    return parts[1];
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "0.0.0";
    }

    private String getCarafeJarPath() {
        try {
            String path = CarafeGUI.class.getProtectionDomain().getCodeSource().getLocation().toURI().getPath();
            if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                if (path.startsWith("/") && path.length() > 2 && path.charAt(2) == ':') {
                    path = path.substring(1);
                }
            }
            if (path.endsWith(".jar")) {
                return path;
            }
            File targetDir = new File("target");
            if (targetDir.exists()) {
                File[] jars = targetDir.listFiles((dir, name) -> name.startsWith("carafe") && name.endsWith(".jar"));
                if (jars != null && jars.length > 0) {
                    return jars[0].getAbsolutePath();
                }
            }
            return "carafe.jar";
        } catch (Exception e) {
            return "carafe.jar";
        }
    }

    private void executeCommand(CmdTask command) {
        isRunning = true;
        runButton.setEnabled(false);
        stopButton.setEnabled(true);
        progressBar.setIndeterminate(true);
        progressBar.setString(command.task_description);
        timeUsageMap.clear();

        if (tabbedPane != null) {
            SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(4));
        }

        Object selectedPython = pythonPathCombo.getSelectedItem();
        String pythonPath = selectedPython != null ? selectedPython.toString().trim() : "";
        if (!pythonPath.isEmpty()) {
            prefs.put(PREF_PYTHON_PATH, pythonPath);
        }

        logToConsole("\n========================================\n");
        logToConsole("Starting Carafe...\n");
        if (!pythonPath.isEmpty()) {
            logToConsole("Python: " + pythonPath + "\n");
        }
        logToConsole("Command: " + command.cmd + "\n");
        logToConsole("========================================\n\n");
        long start = System.nanoTime();
        executor = Executors.newSingleThreadExecutor();
        executor.submit(() -> {
            try {
                ProcessBuilder pb = new ProcessBuilder();
                if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                    pb.command("cmd", "/c", command.cmd);
                } else {
                    pb.command("bash", "-c", command.cmd);
                }
                pb.redirectErrorStream(true);

                if (!pythonPath.isEmpty()) {
                    java.util.Map<String, String> env = pb.environment();
                    File pythonFile = new File(pythonPath);
                    String pythonDir = pythonFile.isFile() ? pythonFile.getParent() : pythonPath;
                    if (pythonDir != null) {
                        String pathSeparator = System.getProperty("os.name").toLowerCase().contains("windows") ? ";"
                                : ":";
                        String currentPath = env.getOrDefault("PATH", env.getOrDefault("Path", ""));
                        String newPath = pythonDir + pathSeparator + currentPath;
                        env.put("PATH", newPath);
                        if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                            env.put("Path", newPath);
                        }
                    }
                }

                currentProcess = pb.start();

                BufferedReader reader = new BufferedReader(new InputStreamReader(currentProcess.getInputStream()));
                String line;
                while ((line = reader.readLine()) != null) {
                    final String output = line;
                    SwingUtilities.invokeLater(() -> {
                        logToConsole(output + "\n");
                        consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
                    });
                }

                int exitCode = currentProcess.waitFor();
                SwingUtilities.invokeLater(() -> {
                    if (exitCode == 0) {
                        long end = System.nanoTime();
                        double minutes = (end - start) / 1e9 / 60.0;
                        String key = "01. " + command.task_name + " - " + command.task_description;
                        timeUsageMap.put(key, minutes);
                        logToConsole("\n[SUCCESS] Carafe completed successfully!\n");
                        progressBar.setString("Completed");
                        logToConsole("\n[SUMMARY] Step durations (min):\n");
                        for (java.util.Map.Entry<String, Double> e : timeUsageMap.entrySet()) {
                            logToConsole(" - " + e.getKey() + " : " + String.format("%.2f", e.getValue()) + "\n");
                        }
                    } else {
                        logToConsole("\n[ERROR] Carafe exited with code: " + exitCode + "\n");
                        progressBar.setString("Failed");
                    }
                    finishExecution();
                });

            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    logToConsole("\n[ERROR] Error: " + e.getMessage() + "\n");
                    progressBar.setString("Error");
                    finishExecution();
                });
            }
        });
    }

    private void stopCarafe() {
        if (currentProcess != null && currentProcess.isAlive()) {
            currentProcess.descendants().forEach(ProcessHandle::destroyForcibly);
            currentProcess.destroyForcibly();
            try {
                currentProcess.waitFor(2, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            logToConsole("\n[STOPPED] Process stopped by user.\n");
        }
        finishExecution();
    }

    private void finishExecution() {
        isRunning = false;

        setInputsFrozen(false);

        runButton.setEnabled(true);
        stopButton.setEnabled(false);
        progressBar.setIndeterminate(false);
        if (executor != null) {
            executor.shutdown();
        }

        if (logWriter != null) {
            try {
                logWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            logWriter = null;
        }
    }

    private boolean validateInputs(int workflowIndex) {
        java.util.List<String> errors = new java.util.ArrayList<>();

        // Helper to check executables
        java.util.function.Consumer<String> checkPython = (label) -> {
            Object py = pythonPathCombo.getSelectedItem();
            String path = (py != null) ? py.toString().trim() : "";
            if (path.isEmpty())
                errors.add("- Python executable is not specified.");
            else if (!new File(path).exists())
                errors.add("- Python executable not found.");
        };

        java.util.function.Consumer<String> checkDiann = (label) -> {
            Object d = diannPathCombo.getSelectedItem();
            String path = (d != null) ? d.toString().trim() : "";
            if (path.isEmpty())
                errors.add("- DIA-NN executable is not specified.");
            else if (!new File(path).exists())
                errors.add("- DIA-NN executable not found.");
        };

        java.util.function.Consumer<String> checkMsConvert = (label) -> {
            // MSConvert must be specified
            String path = "";
            if (msConvertPathCombo != null) {
                Object s = msConvertPathCombo.getSelectedItem();
                if (s != null && !s.toString().trim().isEmpty())
                    path = s.toString().trim();
            }
            if (path.isEmpty()) {
                errors.add("- MSConvert executable is not specified.");
            } else if (!new File(path).exists()) {
                errors.add("- MSConvert executable not found: " + path);
            }
        };

        java.util.function.Consumer<String> checkOutDir = (label) -> {
            String outDir = outputDirField.getText().trim();
            if (outDir.isEmpty())
                errors.add("- Output directory is not specified.");
            else {
                File outDirFile = new File(outDir);
                if (outDirFile.exists()) {
                    if (!outDirFile.isDirectory())
                        errors.add("- Output path exists but is not a directory.");
                    else if (!outDirFile.canWrite())
                        errors.add("- Output directory is not writable.");
                } else {
                    File parent = outDirFile.getParentFile();
                    if (parent != null && !parent.canWrite())
                        errors.add("- Cannot create output directory (parent not writable).");
                }
            }
        };

        // Gather effective inputs
        java.util.List<String> effectiveTrainFiles = new ArrayList<>();
        if (!trainMsFiles.isEmpty())
            effectiveTrainFiles.addAll(trainMsFiles);
        else if (!trainMsFileField.getText().trim().isEmpty())
            effectiveTrainFiles.add(trainMsFileField.getText().trim());

        java.util.List<String> effectiveProjectFiles = new ArrayList<>();
        if (!projectMsFiles.isEmpty())
            effectiveProjectFiles.addAll(projectMsFiles);
        else if (!projectMsFileField.getText().trim().isEmpty())
            effectiveProjectFiles.add(projectMsFileField.getText().trim());

        boolean hasRaw = false;
        // Check for raw files in training entries
        for (String p : effectiveTrainFiles) {
            String low = p.toLowerCase();
            if (low.endsWith(".raw")) {
                hasRaw = true;
                break;
            }
            File f = new File(p);
            if (f.isDirectory()) {
                File[] raws = f.listFiles((d, n) -> n.toLowerCase().endsWith(".raw"));
                if (raws != null && raws.length > 0) {
                    hasRaw = true;
                    break;
                }
            }
        }
        // Check for raw files in project entries if applicable to workflow
        if (workflowIndex == 2 && !hasRaw) {
            for (String p : effectiveProjectFiles) {
                String low = p.toLowerCase();
                if (low.endsWith(".raw")) {
                    hasRaw = true;
                    break;
                }
                File f = new File(p);
                if (f.isDirectory()) {
                    File[] raws = f.listFiles((d, n) -> n.toLowerCase().endsWith(".raw"));
                    if (raws != null && raws.length > 0) {
                        hasRaw = true;
                        break;
                    }
                }
            }
        }

        switch (workflowIndex) {
            case 0: // Library Generation using FASTA
                // 1. Train MS File
                if (effectiveTrainFiles.isEmpty())
                    errors.add("- No Training MS data files selected.");
                else
                    for (String p : effectiveTrainFiles)
                        if (!new File(p).exists()) {
                            errors.add("- Training MS file not found: " + p);
                            break;
                        }

                // 2. Train Protein DB
                String trainDb = trainDbFileField.getText().trim();
                if (trainDb.isEmpty())
                    errors.add("- Training protein database (FASTA) is not specified.");
                else if (!new File(trainDb).exists())
                    errors.add("- Training protein database file not found.");

                // 3. Library Protein DB
                String libDb = libraryDbFileField.getText().trim();
                if (libDb.isEmpty())
                    errors.add("- Library protein database (FASTA) is not specified.");
                else if (!new File(libDb).exists())
                    errors.add("- Library protein database file not found.");

                // 4. Output Directory
                checkOutDir.accept(null);

                // 5. Python Executable
                checkPython.accept(null);

                // 6. DIA-NN Executable
                checkDiann.accept(null);

                // 7. MSConvert (if raw)
                if (hasRaw)
                    checkMsConvert.accept(null);
                break;

            case 1: // Library Refinement

                // 1. Train MS Files (for refinement usage)
                if (effectiveTrainFiles.isEmpty())
                    errors.add("- No MS data files selected for refinement.");
                else
                    for (String p : effectiveTrainFiles)
                        if (!new File(p).exists()) {
                            errors.add("- Train MS File(s) not found: " + p);
                            break;
                        }

                // 2. DIA-NN Report
                String report = diannReportFileField.getText().trim();
                if (report.isEmpty())
                    errors.add("- DIA-NN report file is not specified.");
                else if (!new File(report).exists())
                    errors.add("- DIA-NN report file not found.");

                // 3. Library Protein DB
                String libDbRef = libraryDbFileField.getText().trim();
                if (libDbRef.isEmpty())
                    errors.add("- Library protein database (FASTA) is not specified.");
                else if (!new File(libDbRef).exists())
                    errors.add("- Library protein database file not found.");

                // 4. Output Directory
                checkOutDir.accept(null);

                // 5. Python
                checkPython.accept(null);

                // 6. DIA-NN (Not required per user req, but double check if logic changed?
                // User said "workflow 1 does not need DIA-NN", so skipping checkDiann)

                // 7. MSConvert (if raw)
                if (hasRaw)
                    checkMsConvert.accept(null);
                break;

            case 2: // Whole Workflow
                // 1. Train MS Files
                if (effectiveTrainFiles.isEmpty())
                    errors.add("- No Training MS data files selected.");
                else
                    for (String p : effectiveTrainFiles)
                        if (!new File(p).exists()) {
                            errors.add("- Training MS File(s) not found: " + p);
                            break;
                        }

                // 2. Train Protein DB
                String trainDb2 = trainDbFileField.getText().trim();
                if (trainDb2.isEmpty())
                    errors.add("- Training protein database (FASTA) is not specified.");
                else if (!new File(trainDb2).exists())
                    errors.add("- Training protein database file not found.");

                // 3. Project MS Files
                if (effectiveProjectFiles.isEmpty())
                    errors.add("- No Project MS data files selected.");
                else
                    for (String p : effectiveProjectFiles)
                        if (!new File(p).exists()) {
                            errors.add("- Project MS File(s) not found: " + p);
                            break;
                        }

                // 4. Library Protein DB
                String libDb2 = libraryDbFileField.getText().trim();
                if (libDb2.isEmpty())
                    errors.add("- Library protein database (FASTA) is not specified.");
                else if (!new File(libDb2).exists())
                    errors.add("- Library protein database file not found.");

                // 5. Output Directory
                checkOutDir.accept(null);

                // 6. Python
                checkPython.accept(null);

                // 7. DIA-NN
                checkDiann.accept(null);

                // 8. MSConvert (if raw)
                if (hasRaw)
                    checkMsConvert.accept(null);
                break;
        }

        if (!errors.isEmpty()) {
            StringBuilder msg = new StringBuilder("Please fix the following errors before processing:\n");
            for (String err : errors) {
                msg.append(err).append("\n");
            }
            JOptionPane.showMessageDialog(this, msg.toString(), "Input Validation Failed", JOptionPane.ERROR_MESSAGE);
            return false;
        }

        return true;
    }

    private void setInputsFrozen(boolean frozen) {
        // We only want to freeze the input/settings tabs.
        // Tabs index:
        // 0: Workflow
        // 1: Training Data Generation
        // 2: Model Training
        // 3: Library Generation
        // 4: Console (Do not freeze)

        // Safety check on tab count
        if (tabbedPane != null) {
            int tabCount = tabbedPane.getTabCount();

            // Freeze/Unfreeze first 4 tabs
            for (int i = 0; i < Math.min(tabCount, 4); i++) {
                Component tabComp = tabbedPane.getComponentAt(i);
                if (tabComp instanceof Container) {
                    enableComponents((Container) tabComp, !frozen);
                }
            }
        }

        // Also ensure the run button is toggled (handled in finishExecution/runCarafe
        // but good for safety)
        if (runButton != null)
            runButton.setEnabled(!frozen);
    }

    private void enableComponents(Container container, boolean enable) {
        Component[] components = container.getComponents();
        for (Component component : components) {
            // Do not disable scroll panes, viewports, or scrollbars so scrolling remains
            // possible
            if (component instanceof JScrollPane || component instanceof JViewport || component instanceof JScrollBar) {
                // However, we still need to recurse into them (e.g. into the viewport's view)
                if (component instanceof Container) {
                    enableComponents((Container) component, enable);
                }
                continue;
            }

            component.setEnabled(enable);

            if (component instanceof Container) {
                enableComponents((Container) component, enable);
            }
        }
    }

    private void showHelp() {
        String helpText = """
                Carafe - AI-Powered Spectral Library Generator

                Carafe generates experiment-specific in silico spectral libraries
                using deep learning for DIA data analysis.

                Quick Start:
                1. For fine-tuned library generation:
                   - Provide a peptide detection file (e.g., DIA-NN report.tsv or report.parquet)
                   - Provide MS file(s) in mzML format/Thermo raw/Bruker .d format
                   - Provide protein database (FASTA)
                   - Configure settings and click Run

                For more information, visit:
                https://github.com/Noble-Lab/Carafe

                Citation:
                Wen, B., Hsu, C., Shteynberg, D. et al.
                Carafe enables high quality in silico spectral library generation
                for data-independent acquisition proteomics.
                Nat Commun 16, 9815 (2025).
                """;

        JTextArea textArea = new JTextArea(helpText);
        textArea.setEditable(false);
        textArea.setFont(new Font("Segoe UI", Font.PLAIN, 13));

        JScrollPane scrollPane = new JScrollPane(textArea);
        scrollPane.setPreferredSize(new Dimension(500, 400));

        JOptionPane.showMessageDialog(this, scrollPane, "Carafe Help", JOptionPane.INFORMATION_MESSAGE);
    }

    private static class ScrollablePanel extends JPanel implements Scrollable {
        public ScrollablePanel(java.awt.LayoutManager layout) {
            super(layout);
        }

        @Override
        public Dimension getPreferredScrollableViewportSize() {
            return getPreferredSize();
        }

        @Override
        public int getScrollableUnitIncrement(Rectangle visibleRect, int orientation, int direction) {
            return 16;
        }

        @Override
        public int getScrollableBlockIncrement(Rectangle visibleRect, int orientation, int direction) {
            return 16;
        }

        @Override
        public boolean getScrollableTracksViewportWidth() {
            return true;
        }

        @Override
        public boolean getScrollableTracksViewportHeight() {
            return false;
        }
    }

    private String buildMsConvertCommand(String raw_file, String out_dir) {
        // msconvert.exe --filter "peakPicking true 1-2" --mzML raw_file --outdir
        // out_dir
        // First try to get from UI combo box
        String msConvertExec = "";
        if (msConvertPathCombo != null && msConvertPathCombo.getSelectedItem() != null) {
            msConvertExec = msConvertPathCombo.getSelectedItem().toString().trim();
        }
        // Fall back to preferences
        if (msConvertExec.isEmpty()) {
            msConvertExec = prefs.get(PREF_MSCONVERT_PATH, "");
        }
        // Final fallback to "msconvert" in PATH
        if (msConvertExec.isEmpty()) {
            msConvertExec = "msconvert";
        }
        if (msConvertExec.contains(" "))
            msConvertExec = "\"" + msConvertExec + "\"";

        StringBuilder cmd = new StringBuilder();
        cmd.append(msConvertExec);
        cmd.append(" --filter \"peakPicking true 1-2\" --mzML ");
        cmd.append("\"").append(raw_file).append("\" ");
        cmd.append("-o \"").append(out_dir).append("\"");

        return cmd.toString();
    }

    private String buildMsConvertCommand(java.util.List<String> raw_files, String out_dir) {
        String msConvertExec = "msconvert";
        if (msConvertPathCombo != null) {
            Object selected = msConvertPathCombo.getSelectedItem();
            if (selected != null && !selected.toString().trim().isEmpty()) {
                msConvertExec = selected.toString().trim();
            }
        }

        // If combo was empty/default, try prefs fallback if it differs from default
        if (msConvertExec.equals("msconvert")) {
            String pref = prefs.get(PREF_MSCONVERT_PATH, "");
            if (!pref.isEmpty())
                msConvertExec = pref;
        }

        System.out.println("DEBUG: Using MSConvert executable: " + msConvertExec);

        if (msConvertExec.equalsIgnoreCase("msconvert")) {
            // TODO
        } else {
            // If explicit path, verify existence
            if (!new File(msConvertExec).exists()) {
                JOptionPane.showMessageDialog(this,
                        "The specified MSConvert executable does not exist:\n" + msConvertExec
                                + "\nUsing default 'msconvert' command instead.",
                        "Configuration Warning", JOptionPane.WARNING_MESSAGE);
                msConvertExec = "msconvert";
            }
        }
        if (msConvertExec.contains(" "))
            msConvertExec = "\"" + msConvertExec + "\"";

        StringBuilder cmd = new StringBuilder();
        cmd.append(msConvertExec);
        cmd.append(" --filter \"peakPicking true 1-2\" --mzML ");

        // MSConvert accepts multiple files: msconvert file1 file2 ...
        for (String f : raw_files) {
            cmd.append("\"").append(f).append("\" ");
        }
        cmd.append("-o \"").append(out_dir).append("\"");

        return cmd.toString();
    }

    public static void main(String[] args) {
        System.setProperty("awt.useSystemAAFontSettings", "on");
        System.setProperty("swing.aatext", "true");

        try {
            boolean dark = prefs.getBoolean(PREF_DARK_MODE, false);
            if (dark)
                FlatDarkLaf.setup();
            else
                FlatLightLaf.setup();

            // Enable custom window decorations for FlatLaf to allow hiding the title bar
            // icon
            javax.swing.JFrame.setDefaultLookAndFeelDecorated(true);
            javax.swing.JDialog.setDefaultLookAndFeelDecorated(true);

            // Call defaults directly without creating a dummy window
            customizeUIDefaults();
        } catch (Exception e) {
            e.printStackTrace();
        }

        SwingUtilities.invokeLater(() -> {
            CarafeGUI gui = new CarafeGUI();
            gui.setVisible(true);
        });
    }

    private void chooseFile(String title, int selectionMode, javax.swing.filechooser.FileFilter filter,
            java.util.function.Consumer<File> onFileSelected) {
        setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
        new Thread(() -> {
            try {
                JFileChooser chooser = new JFileChooser();
                String lastDir = prefs.get(PREF_LAST_DIR, System.getProperty("user.home"));
                chooser.setCurrentDirectory(new File(lastDir));
                chooser.setFileSelectionMode(selectionMode);
                if (title != null) {
                    chooser.setDialogTitle(title);
                }
                if (filter != null) {
                    chooser.setFileFilter(filter);
                }
                SwingUtilities.invokeLater(() -> {
                    setCursor(Cursor.getDefaultCursor());
                    if (chooser.showOpenDialog(CarafeGUI.this) == JFileChooser.APPROVE_OPTION) {
                        File f = chooser.getSelectedFile();
                        if (f != null) {
                            onFileSelected.accept(f);
                        }
                    }
                });
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> setCursor(Cursor.getDefaultCursor()));
                ex.printStackTrace();
            }
        }).start();
    }

    private void chooseFiles(String title, String[] extensions, String description,
            java.util.function.Consumer<File[]> onFilesSelected) {
        setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
        new Thread(() -> {
            try {
                JFileChooser chooser = new JFileChooser();
                String lastDir = prefs.get(PREF_LAST_DIR, System.getProperty("user.home"));
                chooser.setCurrentDirectory(new File(lastDir));
                chooser.setMultiSelectionEnabled(true);
                chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                if (title != null) {
                    chooser.setDialogTitle(title);
                }
                if (extensions != null && extensions.length > 0) {
                    javax.swing.filechooser.FileNameExtensionFilter filter = new javax.swing.filechooser.FileNameExtensionFilter(
                            description, extensions);
                    chooser.setFileFilter(filter);
                }
                SwingUtilities.invokeLater(() -> {
                    setCursor(Cursor.getDefaultCursor());
                    if (chooser.showOpenDialog(CarafeGUI.this) == JFileChooser.APPROVE_OPTION) {
                        File[] files = chooser.getSelectedFiles();
                        if (files != null && files.length > 0) {
                            onFilesSelected.accept(files);
                        }
                    }
                });
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> setCursor(Cursor.getDefaultCursor()));
                ex.printStackTrace();
            }
        }).start();
    }

    private void saveParameterScreenshots() {
        String outDir = outputDirField.getText().trim();
        if (outDir.isEmpty()) {
            // Should not happen if workflow ran, but safety check
            return;
        }

        outDir = outDir + File.separator + "parameter_screenshots";
        File dir = new File(outDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        // We want to capture specific tabs that contain parameters.
        // Indices: 0=Workflow, 1=Training Data, 2=Model Training, 3=Library Generation
        String[] tabNames = { "workflow", "training_data", "model_training", "library_generation" };

        logToConsole("\n[INFO] Saving parameter panel screenshots to: " + outDir + "\n");

        // 1. Capture Full Window Screenshot (Current View)
        try {
            java.awt.image.BufferedImage fullWindowImage = new java.awt.image.BufferedImage(
                    this.getWidth(), this.getHeight(), java.awt.image.BufferedImage.TYPE_INT_RGB);
            Graphics2D g2 = fullWindowImage.createGraphics();
            this.validate();
            this.repaint();
            this.print(g2);
            g2.dispose();

            File fullFile = new File(dir, "full_window_capture.png");
            javax.imageio.ImageIO.write(fullWindowImage, "png", fullFile);
            logToConsole(" - Saved: " + fullFile.getName() + " (Full Window)\n");

        } catch (Exception e) {
            logToConsole(" - Failed to save full window screenshot: " + e.getMessage() + "\n");
        }

        // 2. Capture Individual Content Panels (Full Scroll Capture)
        for (int i = 0; i < tabNames.length; i++) {
            if (i >= tabbedPane.getTabCount())
                break;

            Component tabComponent = tabbedPane.getComponentAt(i);

            // The tabs are JScrollPanes wrapping the actual content panels.
            // We need the view component to capture the full size (including off-screen).
            Component view = tabComponent;
            if (tabComponent instanceof JScrollPane sp) {
                view = sp.getViewport().getView();
            }

            if (view != null) {
                try {
                    // Layout buffer if needed, though usually valid by now
                    if (view.getWidth() <= 0 || view.getHeight() <= 0) {
                        continue;
                    }

                    java.awt.image.BufferedImage image = new java.awt.image.BufferedImage(
                            view.getWidth(), view.getHeight(), java.awt.image.BufferedImage.TYPE_INT_RGB);

                    Graphics2D g2 = image.createGraphics();
                    // Fill background explicitly because some panels might be non-opaque or depend
                    // on parent background
                    g2.setColor(view.getBackground());
                    g2.fillRect(0, 0, image.getWidth(), image.getHeight());

                    view.print(g2); // print() is often better than paint() for off-screen full capture
                    g2.dispose();

                    File file = new File(dir, "settings_" + tabNames[i] + ".png");
                    javax.imageio.ImageIO.write(image, "png", file);
                    logToConsole(" - Saved: " + file.getName() + "\n");

                } catch (Exception e) {
                    logToConsole(" - Failed to save screenshot for " + tabNames[i] + ": " + e.getMessage() + "\n");
                }
            }
        }

        logToConsole("\n");
    }
}
