package main.java.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.LinearGradientPaint;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.prefs.Preferences;

import javax.swing.BorderFactory;
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

import org.apache.commons.lang3.StringUtils;

import com.formdev.flatlaf.FlatClientProperties;
import com.formdev.flatlaf.FlatDarkLaf;
import com.formdev.flatlaf.FlatLaf;
import com.formdev.flatlaf.FlatLightLaf;

import ai.djl.util.cuda.CudaUtils;
import main.java.input.CParameter;
import main.java.util.GPUTools;
import main.java.util.GenericUtils;
import main.java.util.PyInstaller;

public class CarafeGUI extends JFrame {

    // global workflow selection
    public static int globalWorkflowIndex = 0;
    private static String carafe_library_directory = "";

    // Brand colors only (keep header/action identity; let FlatLaf handle general UI colors)
    private static final Color PRIMARY_COLOR = new Color(41, 128, 185);
    private static final Color PRIMARY_DARK = new Color(31, 97, 141);
    private static final Color PRIMARY_LIGHT = new Color(52, 152, 219);
    private static final Color ACCENT_COLOR = new Color(46, 204, 113);

    // Layout spacing constants
    private static final int ROW_SPACING = 4;  // Vertical spacing between rows
    private static final int COL_SPACING = 8;  // Horizontal spacing between columns
    private static final Insets DEFAULT_INSETS = new Insets(ROW_SPACING, COL_SPACING, ROW_SPACING, COL_SPACING);

    // Window size constants
    private static final int DEFAULT_WIDTH = 700;   // Default window width
    private static final int DEFAULT_HEIGHT = 750;   // Default window height
    private static final int MIN_WIDTH = 700;       // Minimum window width
    private static final int MIN_HEIGHT = 750;       // Minimum window height
    private static final int COMPONENT_HEIGHT = 32;  // Standard height for input components
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
    private JTextField additionalOptionsField;

    // Input panel rows components for dynamic visibility
    private java.util.List<JComponent> diannReportRowComponents;
    private java.util.List<JComponent> trainMsRowComponents;
    private java.util.List<JComponent> trainDbRowComponents;
    private java.util.List<JComponent> projectMsRowComponents;
    private java.util.List<JComponent> libraryDbRowComponents;
    private java.util.List<JComponent> diannExeRowComponents;
    private JPanel inputFieldsPanel;

    // Training Data Generation settings
    private JSpinner fdrSpinner;
    private JSpinner ptmSiteProbSpinner;
    private JSpinner ptmSiteQvalueSpinner;
    private JSpinner fragTolSpinner;
    private JComboBox<String> fragTolUnitCombo;
    private JCheckBox refineBoundaryCheckbox;
    private JSpinner rtPeakWindowSpinner;
    private JSpinner xicCorSpinner;
    private JSpinner minFragMzSpinner;

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
    private JSpinner maxFragMzSpinner;
    private JSpinner maxFragIonsSpinner;
    private JComboBox<String> libraryFormatCombo;

    // Output console
    private JTextArea consoleArea;
    private JProgressBar progressBar;
    private JButton runButton;
    private JButton stopButton;
    private JTabbedPane tabbedPane;
    private JLabel statusLabel;
    private JScrollPane consoleScrollPane;


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
    private static final String PREF_DARK_MODE = "darkMode";

    public CarafeGUI() {
        setTitle("Carafe - Spectral Library Generator");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setMinimumSize(new Dimension(MIN_WIDTH, MIN_HEIGHT));
        setPreferredSize(new Dimension(DEFAULT_WIDTH, DEFAULT_HEIGHT));
        setResizable(true);

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
        Dimension packedSize = getSize();
        Dimension minSize = getMinimumSize();
        int newWidth = Math.max(packedSize.width, minSize.width);
        int newHeight = Math.max(packedSize.height, minSize.height);
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
    }

    private static Color lafColor(String key, Color fallback) {
        Color c = UIManager.getColor(key);
        return c != null ? c : fallback;
    }

    private void initComponents() {
        setLayout(new BorderLayout());

        // Header
        add(createHeader(), BorderLayout.NORTH);

        // Main content with tabs
        tabbedPane = new JTabbedPane();
        tabbedPane.setFont(new Font("Segoe UI", Font.PLAIN, 13));

        tabbedPane.addTab("1. Workflow", wrapInScrollPane(createInputPanel()));
        tabbedPane.addTab("2. Training Data Generation", wrapInScrollPane(createTrainingDataPanel()));
        tabbedPane.addTab("3. Model Training", wrapInScrollPane(createModelTrainingPanel()));
        tabbedPane.addTab("4. Library Generation", wrapInScrollPane(createLibraryGenerationPanel()));
        tabbedPane.addTab("5. Console", createConsolePanel());

        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 15, 10, 15));
        mainPanel.add(tabbedPane, BorderLayout.CENTER);

        add(mainPanel, BorderLayout.CENTER);

        // Footer with run button
        add(createFooter(), BorderLayout.SOUTH);
    }

    private JScrollPane wrapInScrollPane(JPanel panel) {
        JScrollPane scrollPane = new JScrollPane(panel);
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        scrollPane.setBorder(null);
        scrollPane.getVerticalScrollBar().setUnitIncrement(16);
        scrollPane.getHorizontalScrollBar().setUnitIncrement(16);
        return scrollPane;
    }

    private JPanel createHeader_v1() {
        headerPanel = new JPanel(new BorderLayout());
        headerPanel.setBorder(BorderFactory.createEmptyBorder(20, 25, 20, 25));

        // Logo and title
        headerTitlePanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 0));

        headerIconLabel = new JLabel("C");
        headerIconLabel.setFont(new Font("Segoe UI", Font.BOLD, 42));

        headerTextPanel = new JPanel();
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

        headerPanel.add(headerTitlePanel, BorderLayout.WEST);

        // Right panel with version and dark mode toggle
        headerRightPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 15, 0));

        darkModeToggle = new JToggleButton();
        darkModeToggle.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        darkModeToggle.setFocusPainted(false);
        darkModeToggle.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        darkModeToggle.addActionListener(e -> toggleDarkMode(darkModeToggle.isSelected()));
        headerRightPanel.add(darkModeToggle);

        headerVersionLabel = new JLabel(CParameter.getVersion());
        headerVersionLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        headerRightPanel.add(headerVersionLabel);

        headerPanel.add(headerRightPanel, BorderLayout.EAST);

        // Initialize header appearance based on current theme
        boolean isDark = FlatLaf.isLafDark();
        darkModeToggle.setSelected(isDark);

        return headerPanel;
    }

    private JPanel createHeader_v2() {
        headerPanel = new JPanel(new BorderLayout()) {

            @Override
            protected void paintComponent(Graphics g) {
                int w = getWidth();
                int h = getHeight();
                if (w <= 0 || h <= 0) {
                    super.paintComponent(g);
                    return;
                }

                boolean dark = com.formdev.flatlaf.FlatLaf.isLafDark();

                Color base = UIManager.getColor("Carafe.headerBase");
                if (base == null) base = new Color(0x2F82B7);

                Color top    = dark ? adjust(base, -35) : adjust(base, +70);
                Color mid    = dark ? adjust(base, -18) : adjust(base, +45);
                Color bottom = dark ? adjust(base,  -8) : adjust(base, +25);

                Graphics2D g2 = (Graphics2D) g.create();
                // Paints header background with gradient and highlight
                try {
                    g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                    LinearGradientPaint paint = new LinearGradientPaint(
                            0f, 0f, 0f, (float) h,
                            new float[]{0f, 0.55f, 1f},
                            new Color[]{top, mid, bottom}
                    );
                    g2.setPaint(paint);
                    g2.fillRect(0, 0, w, h);

                    int highlightH = (int) (h * 0.55f);
                    Color hiTop = new Color(255, 255, 255, dark ? 18 : 28);
                    Color hiBot = new Color(255, 255, 255, 0);
                    g2.setPaint(new GradientPaint(0, 0, hiTop, 0, highlightH, hiBot));
                    g2.fillRect(0, 0, w, highlightH);

                } finally {
                    g2.dispose();
                }

                super.paintComponent(g);
            }

            @Override
            public void updateUI() {
                super.updateUI();

                // Updates UI components based on dark mode setting
                SwingUtilities.invokeLater(() -> {
                    boolean dark = com.formdev.flatlaf.FlatLaf.isLafDark();

                    Color base = UIManager.getColor("Carafe.headerBase");
                    if (base == null) base = new Color(0x2F82B7);

                    Color top    = dark ? adjust(base, -35) : adjust(base, +70);
                    Color mid    = dark ? adjust(base, -18) : adjust(base, +45);
                    Color bottom = dark ? adjust(base,  -8) : adjust(base, +25);

                    Color bgSample = mid;

                    Color fgPrimary = pickOnColor(bgSample);              
                    Color fgSecondary = withAlpha(fgPrimary, 215);        
                    Color fgTertiary  = withAlpha(fgPrimary, 190);      

                    if (headerIconLabel != null) headerIconLabel.setForeground(fgPrimary);
                    if (headerTitleLabel != null) headerTitleLabel.setForeground(fgPrimary);
                    if (headerSubtitleLabel != null) headerSubtitleLabel.setForeground(fgSecondary);
                    if (headerVersionLabel != null) headerVersionLabel.setForeground(fgTertiary);

                    if (darkModeToggle != null) {
                        darkModeToggle.setText(dark ? "Light Mode" : "Dark Mode");
                        darkModeToggle.setForeground(fgPrimary);

                        String bg = (fgPrimary.equals(Color.WHITE))
                                ? "rgba(255,255,255,30)"
                                : "rgba(0,0,0,18)";
                        String bd = (fgPrimary.equals(Color.WHITE))
                                ? "rgba(255,255,255,55)"
                                : "rgba(0,0,0,45)";
                        darkModeToggle.putClientProperty(
                                FlatClientProperties.BUTTON_TYPE,
                                FlatClientProperties.BUTTON_TYPE_ROUND_RECT
                        );
                        darkModeToggle.putClientProperty(
                                FlatClientProperties.COMPONENT_ROUND_RECT,
                                Boolean.TRUE
                        );
                        darkModeToggle.setMargin(new Insets(6, 16, 6, 16));
                        darkModeToggle.setFocusPainted(false);
                    }

                    repaint();
                });
            }
        };

        headerPanel.setOpaque(false);
        headerPanel.setBorder(BorderFactory.createEmptyBorder(20, 25, 20, 25));

        // Logo and title
        headerTitlePanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 0));
        headerTitlePanel.setOpaque(false);

        headerIconLabel = new JLabel("C");
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

        headerPanel.add(headerTitlePanel, BorderLayout.WEST);

        // Right panel with version and dark mode toggle
        headerRightPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 15, 0));
        headerRightPanel.setOpaque(false);

        darkModeToggle = new JToggleButton();
        darkModeToggle.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        darkModeToggle.setFocusPainted(false);
        darkModeToggle.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        darkModeToggle.putClientProperty(
                FlatClientProperties.BUTTON_TYPE,
                FlatClientProperties.BUTTON_TYPE_ROUND_RECT
        );
        darkModeToggle.putClientProperty(
                FlatClientProperties.COMPONENT_ROUND_RECT,
                Boolean.TRUE
        );
        darkModeToggle.addActionListener(e -> toggleDarkMode(darkModeToggle.isSelected()));
        headerRightPanel.add(darkModeToggle);

        headerVersionLabel = new JLabel(CParameter.getVersion());
        headerVersionLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        headerRightPanel.add(headerVersionLabel);

        headerPanel.add(headerRightPanel, BorderLayout.EAST);

        boolean isDark = com.formdev.flatlaf.FlatLaf.isLafDark();
        darkModeToggle.setSelected(isDark);
        darkModeToggle.setText(isDark ? "Light Mode" : "Dark Mode");

        headerPanel.updateUI();
        return headerPanel;
    }

    private JPanel createHeader() {
        // Create the panel with custom painting logic
        headerPanel = new JPanel(new BorderLayout()) {
            @Override
            protected void paintComponent(Graphics g) {
                int w = getWidth();
                int h = getHeight();
                if (w <= 0 || h <= 0) {
                    super.paintComponent(g);
                    return;
                }

                boolean dark = FlatLaf.isLafDark();
                Color base = lafColor("Carafe.headerBase", new Color(0x2F82B7));

                // 1. Calculate adaptive gradient colors
                Color top    = dark ? adjust(base, -40) : adjust(base, 60);
                Color mid    = dark ? adjust(base, -20) : adjust(base, 35);
                Color bottom = dark ? adjust(base, -10) : adjust(base, 15);

                Graphics2D g2 = (Graphics2D) g.create();
                try {
                    g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                    // 2. Main Gradient Background
                    LinearGradientPaint paint = new LinearGradientPaint(
                            0f, 0f, 0f, (float) h,
                            new float[]{0f, 0.5f, 1f},
                            new Color[]{top, mid, bottom}
                    );
                    g2.setPaint(paint);
                    g2.fillRect(0, 0, w, h);

                    // 3. Subtle Gloss Highlight
                    int highlightH = (int) (h * 0.5f);
                    Color hiTop = new Color(255, 255, 255, dark ? 15 : 30);
                    Color hiBot = new Color(255, 255, 255, 0);
                    g2.setPaint(new GradientPaint(0, 0, hiTop, 0, highlightH, hiBot));
                    g2.fillRect(0, 0, w, highlightH);
                } finally {
                    g2.dispose();
                }
                super.paintComponent(g);
            }

            @Override
            public void updateUI() {
                super.updateUI();
                updateHeaderForegrounds();
            }
        };

        headerPanel.setOpaque(false);
        headerPanel.setBorder(BorderFactory.createEmptyBorder(20, 25, 20, 25));

        // 4. Layout: Left Side (Logo & Title)
        headerTitlePanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 0));
        headerTitlePanel.setOpaque(false);

        headerIconLabel = new JLabel("C");
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

        // We create the toggle with a custom paintComponent to ENSURE it is rounded and transparent
        darkModeToggle = new JToggleButton() {
            @Override
            protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                try {
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                    boolean dark = FlatLaf.isLafDark();
                    // Semi-transparent background (Darker in dark mode, lighter in light mode)
                    Color bg = dark ? new Color(0, 0, 0, 60) : new Color(255, 255, 255, 60);
                    if (getModel().isRollover()) bg = withAlpha(bg, 100);

                    g2.setColor(bg);
                    g2.fillRoundRect(0, 0, getWidth(), getHeight(), 15, 15);

                    // Subtle border
                    g2.setColor(withAlpha(getForeground(), 80));
                    g2.drawRoundRect(0, 0, getWidth() - 1, getHeight() - 1, 15, 15);
                } finally {
                    g2.dispose();
                }
                super.paintComponent(g);
            }
        };

        darkModeToggle.setContentAreaFilled(false);
        darkModeToggle.setBorderPainted(false);
        darkModeToggle.setOpaque(false);
        darkModeToggle.setFocusPainted(false);
        darkModeToggle.setMargin(new Insets(6, 16, 6, 16));
        darkModeToggle.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        darkModeToggle.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        darkModeToggle.addActionListener(e -> toggleDarkMode(darkModeToggle.isSelected()));

        headerVersionLabel = new JLabel(CParameter.getVersion());
        headerVersionLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));

        headerRightPanel.add(darkModeToggle);
        headerRightPanel.add(headerVersionLabel);

        headerPanel.add(headerTitlePanel, BorderLayout.WEST);
        headerPanel.add(headerRightPanel, BorderLayout.EAST);

        // Sync initial state
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

        if (headerIconLabel != null)     headerIconLabel.setForeground(fgPrimary);
        if (headerTitleLabel != null)    headerTitleLabel.setForeground(fgPrimary);
        if (headerSubtitleLabel != null) headerSubtitleLabel.setForeground(withAlpha(fgPrimary, 210));
        if (headerVersionLabel != null)  headerVersionLabel.setForeground(withAlpha(fgPrimary, 180));

        if (darkModeToggle != null) {
            darkModeToggle.setSelected(dark);
            darkModeToggle.setText(dark ? "Light Mode" : "Dark Mode");
            // This ensures the font is never White-on-White if pickOnColor returns dark for light backgrounds
            darkModeToggle.setForeground(fgPrimary);
        }

        if (headerPanel != null) headerPanel.repaint();
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


    /**
     * C) Correct dark mode switching (robust version):
     * 1) setup() new LaF
     * 2) re-apply UI defaults (A)
     * 3) update UI delegates
     * 4) IMPORTANT: refresh custom components that you manually styled (colors/borders)
     */
    private void toggleDarkMode_v1(boolean isDark) {
        try {
            if (isDark) {
                FlatDarkLaf.setup();
            } else {
                FlatLightLaf.setup();
            }

            customizeUIDefaults();

            // This is the recommended FlatLaf way to refresh whole UI
            FlatLaf.updateUI();

            // Persist preference
            prefs.putBoolean(PREF_DARK_MODE, isDark);

            // Update any custom-painted / manually styled components
            applyThemeToCustomComponents();

            repaint();
        } catch (Exception e) {
            e.printStackTrace();
        }
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

            // 2. Instead of calling the heavy FlatLaf.updateUI(),
            // we update only the root and our specialized components.
            updateHeaderForegrounds();

            applyThemeToCustomComponents();

            // 3. Fast refresh of the window contents
            SwingUtilities.updateComponentTreeUI(this);

            // Persist the preference
            prefs.putBoolean(PREF_DARK_MODE, isDark);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void updateConsoleTheme() {
        Color bg = UIManager.getColor("TextArea.background");
        if (bg == null) bg = UIManager.getColor("TextComponent.background");

        Color fg = UIManager.getColor("TextArea.foreground");
        if (fg == null) fg = UIManager.getColor("TextComponent.foreground");

        Color caret = UIManager.getColor("TextArea.caretForeground");
        if (caret == null) caret = fg;

        consoleArea.setOpaque(true);
        if (bg != null) consoleArea.setBackground(bg);
        if (fg != null) consoleArea.setForeground(fg);
        consoleArea.setCaretColor(caret);

        if (consoleScrollPane.getViewport() != null) {
            consoleScrollPane.getViewport().setOpaque(true);
            if (bg != null) consoleScrollPane.getViewport().setBackground(bg);
        }

        Color spBg = UIManager.getColor("ScrollPane.background");
        if (spBg != null) consoleScrollPane.setBackground(spBg);
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
        refreshStatusLabel();
    }

    private void restyleButtonsRecursively(java.awt.Container root) {
        if (root == null) return;

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

    private void updateInfoCardTheme_v1(InfoCardRef ref) {
        if (ref == null) return;
        boolean dark = FlatLaf.isLafDark();

        Color bg = dark ? new Color(45, 55, 65) : new Color(232, 245, 253);
        Color bd = dark ? new Color(90, 100, 110) : new Color(41, 128, 185, 100);

        ref.card.setBackground(bg);
        ref.card.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(bd),
                BorderFactory.createEmptyBorder(15, 15, 15, 15)
        ));

        if (ref.titleLabel != null) {
            ref.titleLabel.setForeground(dark ? lafColor("Label.foreground", Color.WHITE) : PRIMARY_COLOR);
        }

        if (ref.contentArea != null) {
            ref.contentArea.setBackground(bg);
            ref.contentArea.setForeground(lafColor("Label.foreground", dark ? new Color(220, 220, 220) : new Color(40, 40, 40)));
            ref.contentArea.setCaretColor(ref.contentArea.getForeground());
        }
    }


    private void updateInfoCardTheme(InfoCardRef ref) {
        boolean dark = FlatLaf.isLafDark();

        Color bg      = dark ? new Color(0x2B333A) : new Color(0xE7F3FF);
        Color border  = dark ? new Color(0x55616B) : new Color(0x8BBBE6);
        Color titleFg = dark ? new Color(0x7CC7FF) : new Color(0x2A78B8);
        Color textFg  = dark ? new Color(0xD6DEE6) : UIManager.getColor("Label.foreground");

        // card
        ref.card.setOpaque(true);
        ref.card.setBackground(bg);

        ref.card.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(border, 1, false),
                BorderFactory.createEmptyBorder(16, 16, 16, 16)
        ));

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
        wgbc.anchor = GridBagConstraints.WEST;

        wgbc.gridx = 0; wgbc.gridy = 0; wgbc.weightx = 0;
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
        wgbc.gridx = 1; wgbc.weightx = 1; wgbc.gridwidth = 2;
        workflowPanel.add(workflowCombo, wgbc);

        panel.add(workflowPanel, BorderLayout.NORTH);

        // Center section: Dynamic input fields
        inputFieldsPanel = new JPanel(new GridBagLayout());

        int gridy = 0;

        trainMsRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Train MS File:",
                trainMsFileField = createTextField("Path to mzML file or folder for training"),
                createMsButtonsPanel(trainMsFileField));

        diannReportRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "DIA-NN Report:",
                diannReportFileField = createTextField("Path to DIA-NN report.tsv or report.parquet"),
                createBrowseButton(diannReportFileField, "DIA-NN Report", new String[]{"tsv", "parquet"}));

        trainDbRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Train Protein Database:",
                trainDbFileField = createTextField("Path to protein FASTA for training"),
                createBrowseButton(trainDbFileField, "FASTA Files", new String[]{"fasta", "fa"}));

        projectMsRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Project MS File:",
                projectMsFileField = createTextField("Path to mzML file or folder for project"),
                createMsButtonsPanel(projectMsFileField));

        libraryDbRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Library Protein Database:",
                libraryDbFileField = createTextField("Path to protein FASTA for library generation"),
                createBrowseButton(libraryDbFileField, "FASTA Files", new String[]{"fasta", "fa"}));

        addInputRowToPanel(inputFieldsPanel, gridy++, "Output Directory:",
                outputDirField = createTextField("Path to output directory"),
                createFolderButton(outputDirField));

        addInputRowToPanel(inputFieldsPanel, gridy++, "Python Executable:",
                pythonPathCombo = createPythonComboBox(),
                createPythonBrowseButton());

        diannExeRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "DIA-NN Executable:",
                diannPathCombo = createDiannComboBox(),
                createDiannBrowseButton());

        addInputRowToPanel(inputFieldsPanel, gridy++, "Additional Options:",
                additionalOptionsField = createTextField("Additional command line options"),
                null);

        JPanel infoWrapper = new JPanel(new BorderLayout());
        infoWrapper.add(createInfoCard(
                "Workflow Guide",
                "Workflow 1: Generate spectral library by running DIA-NN search first\n" +
                        "  - Requires: Train MS files, Train database, Library database\n\n" +
                        "Workflow 2: Generate spectral library from existing DIA-NN results\n" +
                        "  - Requires: DIA-NN report file, Train MS files, Library database\n\n" +
                        "Workflow 3: Complete DIA analysis pipeline\n" +
                        "  - Requires: Train MS, Project MS, both databases"
        ), BorderLayout.CENTER);

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

    private java.util.List<JComponent> addInputRowToPanel(JPanel container, int gridy, String labelText, JComponent inputField, JComponent buttonComponent) {
        java.util.List<JComponent> rowComponents = new java.util.ArrayList<>();
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridy = gridy;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(ROW_SPACING, COL_SPACING, ROW_SPACING, COL_SPACING);
        gbc.anchor = GridBagConstraints.WEST;

        gbc.gridx = 0; gbc.weightx = 0;
        JLabel label = createLabel(labelText);
        container.add(label, gbc);
        rowComponents.add(label);

        gbc.gridx = 1; gbc.weightx = 1;
        if (buttonComponent == null) {
            gbc.gridwidth = 2;
        }
        container.add(inputField, gbc);
        rowComponents.add(inputField);
        gbc.gridwidth = 1;

        if (buttonComponent != null) {
            gbc.gridx = 2; gbc.weightx = 0;
            container.add(buttonComponent, gbc);
            rowComponents.add(buttonComponent);
        }

        return rowComponents;
    }

    private JPanel createMsButtonsPanel(JTextField targetField) {
        JPanel msButtonsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 2, 0));
        msButtonsPanel.add(createBrowseButton(targetField, "mzML Files", new String[]{"mzML"}));
        msButtonsPanel.add(createFolderButton(targetField));
        return msButtonsPanel;
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
            }
            case 1 -> { // Workflow 2
                setVisible(diannReportRowComponents, true);
                setVisible(trainMsRowComponents, true);
                setVisible(trainDbRowComponents, false);
                setVisible(projectMsRowComponents, false);
                setVisible(libraryDbRowComponents, true);
                setVisible(diannExeRowComponents, false);
            }
            case 2 -> { // Workflow 3
                setVisible(diannReportRowComponents, false);
                setVisible(trainMsRowComponents, true);
                setVisible(trainDbRowComponents, true);
                setVisible(projectMsRowComponents, true);
                setVisible(libraryDbRowComponents, true);
                setVisible(diannExeRowComponents, true);
            }
        }

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

        gbc.gridx = 0; gbc.gridy = 0; gbc.weightx = 0;
        panel.add(createLabel("False Discovery Rate:"), gbc);

        fdrSpinner = createDoubleSpinner(0.01, 0.001, 0.1, 0.005);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(fdrSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 1; gbc.weightx = 0;
        panel.add(createLabel("PTM Site Probability:"), gbc);

        ptmSiteProbSpinner = createDoubleSpinner(0.75, 0.0, 1.0, 0.05);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(ptmSiteProbSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 2; gbc.weightx = 0;
        panel.add(createLabel("PTM Site Q-value:"), gbc);

        ptmSiteQvalueSpinner = createDoubleSpinner(0.01, 0.001, 0.1, 0.005);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(ptmSiteQvalueSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 3; gbc.weightx = 0;
        panel.add(createLabel("Fragment Ion Mass Tolerance:"), gbc);

        fragTolSpinner = createSpinner(20, 1, 100, 1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(fragTolSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 4; gbc.weightx = 0;
        panel.add(createLabel("Fragment Ion Mass Tolerance Units:"), gbc);

        String[] tolUnits = {"ppm", "Da"};
        fragTolUnitCombo = new JComboBox<>(tolUnits);
        styleComboBox(fragTolUnitCombo);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(fragTolUnitCombo, gbc);

        gbc.gridx = 0; gbc.gridy = 5; gbc.weightx = 0;
        panel.add(createLabel("Refine Peak Boundaries:"), gbc);

        refineBoundaryCheckbox = createCheckBox("", true);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(refineBoundaryCheckbox, gbc);

        gbc.gridx = 0; gbc.gridy = 6; gbc.weightx = 0;
        panel.add(createLabel("RT Peak Window:"), gbc);

        rtPeakWindowSpinner = createSpinner(3, 1, 10, 1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(rtPeakWindowSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 7; gbc.weightx = 0;
        panel.add(createLabel("XIC Correlation:"), gbc);

        xicCorSpinner = createDoubleSpinner(0.8, 0.0, 1.0, 0.05);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(xicCorSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 8; gbc.weightx = 0;
        panel.add(createLabel("Minimum Fragment m/z:"), gbc);

        minFragMzSpinner = createSpinner(200, 50, 500, 10);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(minFragMzSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 9; gbc.gridwidth = 2; gbc.weighty = 1;
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

        gbc.gridx = 0; gbc.gridy = 0; gbc.weightx = 0;
        panel.add(createLabel("Model Type:"), gbc);

        String[] modes = {"general", "phosphorylation"};
        modeCombo = new JComboBox<>(modes);
        styleComboBox(modeCombo);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(modeCombo, gbc);

        gbc.gridx = 0; gbc.gridy = 1; gbc.weightx = 0;
        panel.add(createLabel("Normalized Collision Energy:"), gbc);
        nceField = createTextField("e.g., 27 or auto");
        nceField.setText("auto");
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(nceField, gbc);

        gbc.gridx = 0; gbc.gridy = 2; gbc.weightx = 0;
        panel.add(createLabel("MS Instrument Type:"), gbc);

        String[] msInstruments = {"auto", "QE", "Lumos", "timsTOF", "SciexTOF", "ThermoTOF"};
        msInstrumentField = new JComboBox<>(msInstruments);
        msInstrumentField.setEditable(false);
        styleComboBox(msInstrumentField);
        msInstrumentField.setSelectedItem("auto");
        msInstrumentField.setToolTipText("Select MS instrument (one of auto, QE, Lumos, timsTOF, SciexTOF, ThermoTOF)");
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(msInstrumentField, gbc);

        gbc.gridx = 0; gbc.gridy = 3; gbc.weightx = 0;
        panel.add(createLabel("Computational Device:"), gbc);

        String[] devices = {"auto", "gpu", "cpu"};
        deviceCombo = new JComboBox<>(devices);
        deviceCombo.setEditable(false);
        styleComboBox(deviceCombo);
        deviceCombo.setSelectedItem("auto");
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(deviceCombo, gbc);

        gbc.gridx = 0; gbc.gridy = 4; gbc.gridwidth = 2; gbc.weightx = 1;
        gbc.insets = new Insets(20, 8, 8, 8);
        panel.add(createInfoCard(
                "Model Training Tips",
                "- GPU mode is recommended for faster training (requires CUDA-compatible GPU)\n" +
                        "- If GPU is not available, the software will automatically fall back to CPU\n" +
                        "- NCE and MS Instrument are optional for fine-tuning (learned from data)\n" +
                        "- Use 'phosphorylation' mode for phosphopeptide analysis"
        ), gbc);

        gbc.gridy = 5; gbc.weighty = 1;
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

        gbc.gridx = 0; gbc.gridy = 0; gbc.weightx = 0;
        panel.add(createLabel("Enzyme:"), gbc);

        String[] enzymes = {
                "1:Trypsin (default)",
                "2:Trypsin (no P rule)",
                "3:Arg-C",
                "4:Arg-C (no P rule)",
                "5:Arg-N",
                "6:Glu-C",
                "7:Lys-C",
                "0:Non enzyme"
        };
        enzymeCombo = new JComboBox<>(enzymes);
        styleComboBox(enzymeCombo);
        enzymeCombo.setSelectedIndex(1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(enzymeCombo, gbc);

        gbc.gridx = 0; gbc.gridy = 1; gbc.weightx = 0;
        panel.add(createLabel("Missed Cleavages:"), gbc);

        missCleavageSpinner = createSpinner(1, 0, 5, 1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(missCleavageSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 2; gbc.weightx = 0;
        panel.add(createLabel("Fixed Modification Available:"), gbc);

        String[] fixMods = {
                "1 - Carbamidomethylation (C) [57.02]",
                "0 - No fixed modification"
        };
        fixModAvailableCombo = new JComboBox<>(fixMods);
        styleComboBox(fixModAvailableCombo);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(fixModAvailableCombo, gbc);

        gbc.gridx = 0; gbc.gridy = 3; gbc.weightx = 0;
        panel.add(createLabel("Fixed Modifications Selected:"), gbc);

        fixModSelectedField = createTextField("e.g., 1");
        fixModSelectedField.setText("1");
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(fixModSelectedField, gbc);

        gbc.gridx = 0; gbc.gridy = 4; gbc.weightx = 0;
        panel.add(createLabel("Variable Modifications Available:"), gbc);

        String[] varMods = {
                "2 - Oxidation (M) [15.99]",
                "7,8,9 - Phosphorylation (STY)",
                "2,7,8,9 - Oxidation (M) + Phosphorylation (STY)",
                "0 - No variable modification"
        };
        varModAvailableCombo = new JComboBox<>(varMods);
        styleComboBox(varModAvailableCombo);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(varModAvailableCombo, gbc);

        gbc.gridx = 0; gbc.gridy = 5; gbc.weightx = 0;
        panel.add(createLabel("Variable Modifications Selected:"), gbc);

        varModSelectedField = createTextField("e.g., 0 or 2");
        varModSelectedField.setText("0");
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(varModSelectedField, gbc);

        gbc.gridx = 0; gbc.gridy = 6; gbc.weightx = 0;
        panel.add(createLabel("Maximum Variable Modifications:"), gbc);

        maxVarSpinner = createSpinner(1, 0, 5, 1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(maxVarSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 7; gbc.weightx = 0;
        panel.add(createLabel("Clip N-Terminal Methionine:"), gbc);

        clipNmCheckbox = createCheckBox("", true);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(clipNmCheckbox, gbc);

        gbc.gridx = 0; gbc.gridy = 8; gbc.weightx = 0;
        panel.add(createLabel("Minimum Peptide Length:"), gbc);

        minLengthSpinner = createSpinner(7, 1, 50, 1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(minLengthSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 9; gbc.weightx = 0;
        panel.add(createLabel("Maximum Peptide Length:"), gbc);

        maxLengthSpinner = createSpinner(35, 1, 100, 1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(maxLengthSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 10; gbc.weightx = 0;
        panel.add(createLabel("Minimum Peptide m/z:"), gbc);

        minPepMzSpinner = createSpinner(400, 100, 2000, 50);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(minPepMzSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 11; gbc.weightx = 0;
        panel.add(createLabel("Maximum Peptide m/z:"), gbc);

        maxPepMzSpinner = createSpinner(1000, 100, 3000, 50);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(maxPepMzSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 12; gbc.weightx = 0;
        panel.add(createLabel("Minimum Peptide Charge:"), gbc);

        minPepChargeSpinner = createSpinner(2, 1, 10, 1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(minPepChargeSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 13; gbc.weightx = 0;
        panel.add(createLabel("Maximum Peptide Charge:"), gbc);

        maxPepChargeSpinner = createSpinner(4, 1, 10, 1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(maxPepChargeSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 14; gbc.weightx = 0;
        panel.add(createLabel("Minimum Fragment m/z:"), gbc);

        // Note: This UI shows a second copy; command currently uses minFragMzSpinner (from training tab),
        // keeping behavior identical to your original file.
        JSpinner minFragMzSpinner2 = createSpinner(200, 50, 500, 10);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(minFragMzSpinner2, gbc);

        gbc.gridx = 0; gbc.gridy = 15; gbc.weightx = 0;
        panel.add(createLabel("Maximum Fragment m/z:"), gbc);

        maxFragMzSpinner = createSpinner(1800, 500, 3000, 50);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(maxFragMzSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 16; gbc.weightx = 0;
        panel.add(createLabel("Maximum Number of Fragment Ions:"), gbc);

        maxFragIonsSpinner = createSpinner(20, 1, 100, 1);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(maxFragIonsSpinner, gbc);

        gbc.gridx = 0; gbc.gridy = 17; gbc.weightx = 0;
        panel.add(createLabel("Spectral Library Format:"), gbc);

        String[] formats = {"DIA-NN", "Skyline", "EncyclopeDIA", "mzSpecLib"};
        libraryFormatCombo = new JComboBox<>(formats);
        styleComboBox(libraryFormatCombo);
        gbc.gridx = 1; gbc.weightx = 1;
        panel.add(libraryFormatCombo, gbc);

        gbc.gridx = 0; gbc.gridy = 18; gbc.gridwidth = 2; gbc.weighty = 1;
        panel.add(Box.createVerticalGlue(), gbc);

        return panel;
    }

    private JPanel createConsolePanel() {
        JPanel panel = new JPanel(new BorderLayout());
        Color border = lafColor("Component.borderColor", lafColor("Separator.foreground", new Color(128, 128, 128)));
        panel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(border),
                BorderFactory.createEmptyBorder(10, 10, 10, 10)
        ));

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
        consoleArea.setLineWrap(true);
        consoleArea.setWrapStyleWord(true);

        consoleScrollPane   = new JScrollPane(consoleArea);
        consoleScrollPane .setBorder(BorderFactory.createLineBorder(border));
        consoleScrollPane .setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        panel.add(consoleScrollPane , BorderLayout.CENTER);

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
                new String[]{"Copy to Clipboard", "Close"}, "Close");

        if (result == 0) {
            java.awt.datatransfer.StringSelection selection = new java.awt.datatransfer.StringSelection(command);
            java.awt.Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);
            JOptionPane.showMessageDialog(this, "Command copied to clipboard!", "Copied", JOptionPane.INFORMATION_MESSAGE);
        }
    }

    private void refreshStatusLabel() {
        if (statusLabel == null) return;

        String py = "Not Set";
        try {
            if (pythonPathCombo != null && pythonPathCombo.getSelectedItem() != null) {
                py = pythonPathCombo.getSelectedItem().toString();
            }
        } catch (Exception ignored) {}

        statusLabel.setText("Ready | GPU: " + cachedGpuStatus + " | Python: " + py);
    }

    private void updateGpuStatusAsync() {
        if (pythonPathCombo == null) return;

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
            JFileChooser chooser = new JFileChooser();
            String lastDir = prefs.get(PREF_LAST_DIR, System.getProperty("user.home"));
            chooser.setCurrentDirectory(new File(lastDir));
            chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
            if (extensions != null && extensions.length > 0) {
                FileNameExtensionFilter filter = new FileNameExtensionFilter(description, extensions);
                chooser.setFileFilter(filter);
            }
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                File selectedFile = chooser.getSelectedFile();
                targetField.setText(selectedFile.getAbsolutePath());
                prefs.put(PREF_LAST_DIR, selectedFile.getParent());
            }
        });
        return button;
    }

    private JButton createFolderButton(JTextField targetField) {
        JButton button = new JButton("Folder");
        styleButton(button);
        button.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser();
            String lastDir = prefs.get(PREF_LAST_DIR, System.getProperty("user.home"));
            chooser.setCurrentDirectory(new File(lastDir));
            chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                File selectedDir = chooser.getSelectedFile();
                targetField.setText(selectedDir.getAbsolutePath());
                prefs.put(PREF_LAST_DIR, selectedDir.getAbsolutePath());
            }
        });
        return button;
    }

    private JPanel createPythonBrowseButton() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));

        JButton browse = new JButton("Browse");
        styleButton(browse);
        browse.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser();
            String defaultDir = System.getProperty("os.name").toLowerCase().contains("windows") ? "C:\\" : "/usr/bin";
            String lastDir = prefs.get(PREF_LAST_DIR, defaultDir);
            chooser.setCurrentDirectory(new File(lastDir));
            chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
            chooser.setDialogTitle("Select Python Executable");
            if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                chooser.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("Executable Files", "exe"));
            }
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
                        if (tabbedPane != null) tabbedPane.setSelectedIndex(Math.max(0, tabbedPane.getTabCount() - 1));
                        progressBar.setIndeterminate(true);
                        progressBar.setString("Python installation...");
                        consoleArea.append("\n[INSTALL] Python installation started...\n");
                        consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
                    });

                    java.nio.file.Path logFile = installRoot.resolve("logs").resolve("install.log");
                    AtomicBoolean installDone = new AtomicBoolean(false);
                    Thread tailer = new Thread(() -> {
                        try {
                            while (!installDone.get() && !java.nio.file.Files.exists(logFile)) {
                                Thread.sleep(200);
                            }
                            if (!java.nio.file.Files.exists(logFile)) return;
                            try (RandomAccessFile raf = new RandomAccessFile(logFile.toFile(), "r")) {
                                long pointer = 0;
                                while (!installDone.get()) {
                                    long len = raf.length();
                                    if (len > pointer) {
                                        raf.seek(pointer);
                                        String line;
                                        while ((line = raf.readLine()) != null) {
                                            final String decoded = new String(line.getBytes("ISO-8859-1"), StandardCharsets.UTF_8);
                                            SwingUtilities.invokeLater(() -> {
                                                consoleArea.append(decoded + "\n");
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
                                        final String decoded = new String(line.getBytes("ISO-8859-1"), StandardCharsets.UTF_8);
                                        SwingUtilities.invokeLater(() -> {
                                            consoleArea.append(decoded + "\n");
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
                    try { tailer.join(500); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); }

                    final String installedPath = py_path;
                    SwingUtilities.invokeLater(() -> {
                        consoleArea.append("[INSTALL] Completed. Python installed at: " + installedPath + "\n");
                        consoleArea.setCaretPosition(consoleArea.getDocument().getLength());

                        boolean found = false;
                        for (int i = 0; i < pythonPathCombo.getItemCount(); i++) {
                            if (pythonPathCombo.getItemAt(i).equals(installedPath)) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) pythonPathCombo.addItem(installedPath);
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
                        consoleArea.append("[INSTALL] Failed: " + msg + "\n");
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

        java.util.List<String> pythonPaths = detectPythonInstallations();
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
                if (basePath == null) continue;
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
            } catch (Exception ignored) {}

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
                if (path == null) continue;
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
            } catch (Exception ignored) {}

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
            } catch (Exception ignored) {}
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
                    } catch (Exception ignored) {}
                }
            } catch (Exception ignored) {}
        }
    }

    private JButton createDiannBrowseButton() {
        JButton button = new JButton("Browse");
        styleButton(button);
        button.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser();
            String defaultDir = System.getProperty("os.name").toLowerCase().contains("windows") ? "C:\\" : "/usr/bin";
            String lastDir = prefs.get(PREF_LAST_DIR, defaultDir);
            chooser.setCurrentDirectory(new File(lastDir));
            chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
            chooser.setDialogTitle("Select DIA-NN Executable");
            if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                chooser.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("Executable Files", "exe"));
            }
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
                    "C:\\Program Files (x86)\\DIA-NN"
            };

            for (String basePath : windowsPaths) {
                if (basePath == null) continue;
                File baseDir = new File(basePath);
                if (baseDir.exists() && baseDir.isDirectory()) {
                    File diannExe = new File(baseDir, "diann.exe");
                    if (diannExe.exists()) {
                        diannPaths.add(diannExe.getAbsolutePath());
                    }
                    File[] subDirs = baseDir.listFiles(File::isDirectory);
                    if (subDirs != null) {
                        for (File subDir : subDirs) {
                            diannExe = new File(subDir, "diann.exe");
                            if (diannExe.exists()) {
                                diannPaths.add(diannExe.getAbsolutePath());
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
            } catch (Exception ignored) {}

        } else {
            String[] unixPaths = {
                    "/usr/local/bin/diann",
                    "/usr/bin/diann",
                    System.getenv("HOME") + "/DIA-NN/diann",
                    "/opt/DIA-NN/diann"
            };

            for (String path : unixPaths) {
                if (path == null) continue;
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
            } catch (Exception ignored) {}
        }

        if (diannPaths.isEmpty()) {
            diannPaths.add(isWindows ? "diann.exe" : "diann");
        }
        return diannPaths;
    }

    private void styleButton_v1(JButton button) {
        button.putClientProperty("carafe.role", "generic");
        button.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        Color border = lafColor("Component.borderColor", lafColor("Separator.foreground", new Color(128, 128, 128)));
        button.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(border),
                BorderFactory.createEmptyBorder(6, 12, 6, 12)
        ));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
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
                BorderFactory.createEmptyBorder(10, 20, 10, 20)
        ));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
    }

    private JButton createPrimaryButton_v1(String text, Color color) {
        JButton button = new JButton(text);
        button.putClientProperty("carafe.role", "primary");
        button.setFont(new Font("Segoe UI", Font.BOLD, 14));
        button.setBackground(color);
        button.setForeground(Color.WHITE);
        button.setBorder(BorderFactory.createEmptyBorder(12, 30, 12, 30));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        button.setOpaque(true);
        return button;
    }

    private JButton createPrimaryButton(String text, Color color) {
        JButton button = new JButton(text);
        button.setFont(new Font("Segoe UI", Font.BOLD, 14));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        button.setMargin(new Insets(12, 30, 12, 30));

        button.putClientProperty("JButton.buttonType", "roundRect");

        // FlatLaf 推荐：用 client property 设置颜色（不会破坏 painter）
        button.putClientProperty("JButton.background", color);
        button.putClientProperty("JButton.foreground", Color.WHITE);

        // 不要 setOpaque(true) / setBorder(...) / setContentAreaFilled(false) 这些
        return button;
    }


    private JButton createSecondaryButton_v1(String text) {
        JButton button = new JButton(text);
        styleSecondaryButton(button);
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

    private JPanel createInfoCard_v1(String title, String content) {
        JPanel card = new JPanel(new BorderLayout());

        JLabel titleLabel = new JLabel(title);
        titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 13));
        titleLabel.setBorder(BorderFactory.createEmptyBorder(0, 0, 8, 0));
        card.add(titleLabel, BorderLayout.NORTH);

        JTextArea contentArea = new JTextArea(content) {
            @Override
            public Dimension getPreferredSize() {
                Dimension d = super.getPreferredSize();
                return new Dimension(100, d.height);
            }
        };
        contentArea.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        contentArea.setEditable(false);
        contentArea.setLineWrap(true);
        contentArea.setWrapStyleWord(true);
        card.add(contentArea, BorderLayout.CENTER);

        InfoCardRef ref = new InfoCardRef(card, titleLabel, contentArea);
        infoCards.add(ref);
        updateInfoCardTheme(ref);

        return card;
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
        if (wf == 1) return buildCarafeCommand();
        return "This workflow runs chained commands. Click Run Carafe to execute.";
    }

    private String buildCarafeCommand() {
        StringBuilder cmd = new StringBuilder();
        boolean exe_launch = false;
        String javaExec = getJavaExecutable();
        if(javaExec.endsWith("java.exe") || javaExec.endsWith("java")) {
            // use as is
        } else if(javaExec.endsWith("Carafe.exe")) {
            // it is likely launched from bundled Carafe.exe
            File javaFile = new File(javaExec);
            // navigate to ../runtime/bin/java.exe
            exe_launch = true;
        }else{
            // fallback to "java" in PATH
            javaExec = "java";
        }
        if (javaExec.contains(" ")) {
            javaExec = '"' + javaExec + '"';
        }
        // get the computer memory and set Xmx accordingly
        int memory_use = (int) Math.ceil(GenericUtils.get_system_memory_available()*0.8);

        if(exe_launch) {
            cmd.append(javaExec).append(" ");
        }else{
            cmd.append(javaExec).append(" -Xmx").append(memory_use).append("G ");

            int javaVersion = GenericUtils.getJavaMajorVersion();
            if (javaVersion >= 18 && javaVersion <= 23) {
                cmd.append("-Djava.security.manager=allow ");
            }
            cmd.append("-jar ");
            String carafeJarPath = getCarafeJarPath();
            cmd.append(carafeJarPath).append(" ");
        }

        int workflow = workflowCombo.getSelectedIndex();

        String libraryDb = libraryDbFileField.getText().trim();
        if (!libraryDb.isEmpty()) {
            cmd.append("-db \"").append(libraryDb).append("\" ");
        }

        String diannReport = diannReportFileField.getText().trim();
        System.out.println("diannReport:" + diannReport);
        if (!diannReport.isEmpty()) {
            cmd.append("-i \"").append(diannReport).append("\" ");
        }

        String trainMsFile = trainMsFileField.getText().trim();
        if (!trainMsFile.isEmpty()) {
            cmd.append("-ms \"").append(trainMsFile).append("\" ");
        }

        String outDir = outputDirField.getText().trim();
        if (!outDir.isEmpty()) {
            carafe_library_directory = outDir + File.separator + "carafe_library";
            cmd.append("-o \"").append(carafe_library_directory).append("\" ");
        }

        cmd.append("-fdr ").append(fdrSpinner.getValue()).append(" ");
        cmd.append("-ptm_site_prob ").append(ptmSiteProbSpinner.getValue()).append(" ");
        cmd.append("-ptm_site_qvalue ").append(ptmSiteQvalueSpinner.getValue()).append(" ");
        cmd.append("-itol ").append(fragTolSpinner.getValue()).append(" ");
        cmd.append("-itolu ").append(fragTolUnitCombo.getSelectedItem()).append(" ");
        if (refineBoundaryCheckbox.isSelected()) cmd.append("-rf ");
        cmd.append("-rf_rt_win ").append(rtPeakWindowSpinner.getValue()).append(" ");
        cmd.append("-cor ").append(xicCorSpinner.getValue()).append(" ");
        cmd.append("-lf_frag_mz_min ").append(minFragMzSpinner.getValue()).append(" ");

        cmd.append("-mode ").append(modeCombo.getSelectedItem()).append(" ");
        String nce = nceField.getText().trim();
        if (!nce.isEmpty()) {
            if (!nce.equalsIgnoreCase("auto")) {
                cmd.append("-nce ").append(nce).append(" ");
            }
        }
        Object msSel = msInstrumentField.getSelectedItem();
        String msInstrument = msSel == null ? "" : msSel.toString().trim();
        if (!msInstrument.isEmpty()) {
            if (!msInstrument.equalsIgnoreCase("auto")) {
                cmd.append("-ms_instrument ").append(msInstrument).append(" ");
            }
        }

        Object deviceSel = deviceCombo.getSelectedItem();
        String device = deviceSel == null ? "auto" : deviceSel.toString().trim();
        if (device.equalsIgnoreCase("auto")) {
            boolean available = "Available".equals(cachedGpuStatus);
            cmd.append("-device ").append(available ? "gpu" : "cpu").append(" ");
        } else {
            cmd.append("-device ").append(device).append(" ");
        }

        String enzyme = ((String) enzymeCombo.getSelectedItem()).split(":")[0];
        cmd.append("-enzyme ").append(enzyme).append(" ");
        cmd.append("-miss_c ").append(missCleavageSpinner.getValue()).append(" ");

        String fixModSelected = fixModSelectedField.getText().trim();
        if (!fixModSelected.isEmpty()) {
            cmd.append("-fixMod ").append(fixModSelected).append(" ");
        }

        String varModSelected = varModSelectedField.getText().trim();
        if (!varModSelected.isEmpty()) {
            cmd.append("-varMod ").append(varModSelected).append(" ");
        }

        cmd.append("-maxVar ").append(maxVarSpinner.getValue()).append(" ");
        if (clipNmCheckbox.isSelected()) cmd.append("-clip_n_m ");
        cmd.append("-minLength ").append(minLengthSpinner.getValue()).append(" ");
        cmd.append("-maxLength ").append(maxLengthSpinner.getValue()).append(" ");
        cmd.append("-min_pep_mz ").append(minPepMzSpinner.getValue()).append(" ");
        cmd.append("-max_pep_mz ").append(maxPepMzSpinner.getValue()).append(" ");
        cmd.append("-min_pep_charge ").append(minPepChargeSpinner.getValue()).append(" ");
        cmd.append("-max_pep_charge ").append(maxPepChargeSpinner.getValue()).append(" ");
        cmd.append("-lf_frag_mz_max ").append(maxFragMzSpinner.getValue()).append(" ");
        cmd.append("-lf_top_n_frag ").append(maxFragIonsSpinner.getValue()).append(" ");
        cmd.append("-lf_type ").append(libraryFormatCombo.getSelectedItem()).append(" ");
        cmd.append("-se DIA-NN ");

        if (!trainMsFile.isEmpty()) {
            cmd.append("-tf all ");
        }

        String additionalOptions = additionalOptionsField.getText().trim();
        if (!additionalOptions.isEmpty()) {
            cmd.append(additionalOptions).append(" ");
        }

        return cmd.toString();
    }

    private String getJavaExecutable() {
        try {
            Optional<String> cmd = java.lang.ProcessHandle.current().info().command();
            if (cmd.isPresent()) {
                return cmd.get();
            }
        } catch (Throwable ignored) {}
        String javaHome = System.getProperty("java.home");
        String sep = System.getProperty("file.separator");
        return javaHome + sep + "bin" + sep + (System.getProperty("os.name").toLowerCase().contains("win") ? "java.exe" : "java");
    }

    private void runCarafe() {
        if (isRunning) {
            JOptionPane.showMessageDialog(this, "A process is already running!", "Warning", JOptionPane.WARNING_MESSAGE);
            return;
        }

        int workflow = workflowCombo.getSelectedIndex();
        switch (workflow) {
            case 0 -> {
                String trainMsFile = trainMsFileField.getText().trim();
                if (trainMsFile.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please specify a training MS/MS file!", "Warning", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                String trainDb = trainDbFileField.getText().trim();
                if (trainDb.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please provide a training protein database file.", "Input Required", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                String libraryDb = libraryDbFileField.getText().trim();
                if (libraryDb.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please provide a library protein database file.", "Input Required", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                String outDir = outputDirField.getText().trim();
                if (outDir.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please specify an output directory!", "Warning", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                String diann_train_dir = outDir + File.separator + "diann_train";
                File diannTrainDirFile = new File(diann_train_dir);
                if (!diannTrainDirFile.exists()) {
                    diannTrainDirFile.mkdirs();
                }
                String diann_cmd = buildDIANNCommand(trainMsFile, "", trainDb, diann_train_dir);
                String diann_report_file = diann_train_dir + File.separator + "report.parquet";

                if (tabbedPane != null) {
                    SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(4));
                }

                executeChainedCommands(new CmdTask[]{new CmdTask(diann_cmd, "DIA-NN", "Run DIA-NN search on the training MS data...")}, () -> {
                    final CmdTask[] commandContainer = new CmdTask[1];
                    try {
                        SwingUtilities.invokeAndWait(() -> {
                            diannReportFileField.setText(diann_report_file);
                            commandContainer[0] = new CmdTask(buildCarafeCommand(), "Carafe", "Run Carafe to generate fine-tuned library ...");
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    return new CmdTask[]{commandContainer[0]};
                });
            }
            case 1 -> {
                String libraryDb = libraryDbFileField.getText().trim();
                if (libraryDb.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please provide a library protein database file.", "Input Required", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                String carafe_cmd = buildCarafeCommand();
                executeCommand(new CmdTask(carafe_cmd, "Carafe", "Run Carafe to generate spectral library..."));
            }
            case 2 -> {
                String trainMsFile = trainMsFileField.getText().trim();
                if (trainMsFile.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please specify a training MS/MS file!", "Warning", JOptionPane.WARNING_MESSAGE);
                    return;
                }

                String projectMsFile = projectMsFileField.getText().trim();
                if (projectMsFile.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please specify a project MS/MS file!", "Warning", JOptionPane.WARNING_MESSAGE);
                    return;
                }

                String trainDb = trainDbFileField.getText().trim();
                if (trainDb.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please provide a training protein database file.", "Input Required", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                String libraryDb = libraryDbFileField.getText().trim();
                if (libraryDb.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please provide a library protein database file.", "Input Required", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                String outDir = outputDirField.getText().trim();
                if (outDir.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "Please specify an output directory!", "Warning", JOptionPane.WARNING_MESSAGE);
                    return;
                }
                String diann_train_dir = outDir + File.separator + "diann_train";
                File diannTrainDirFile = new File(diann_train_dir);
                if (!diannTrainDirFile.exists()) {
                    diannTrainDirFile.mkdirs();
                }

                String diann_cmd = buildDIANNCommand(trainMsFile, "", trainDb, diann_train_dir);
                String diann_report_file = diann_train_dir + File.separator + "report.parquet";

                String diann_project_dir = outDir + File.separator + "diann_project";
                File diannProjectDirFile = new File(diann_project_dir);
                if (!diannProjectDirFile.exists()) {
                    diannProjectDirFile.mkdirs();
                }
                final String carafeLibraryPath = outDir + File.separator + "carafe_library" + File.separator + "SkylineAI_spectral_library.tsv";

                if (tabbedPane != null) {
                    SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(4));
                }

                executeChainedCommands(new CmdTask[]{new CmdTask(diann_cmd, "DIA-NN", "Run DIA-NN search on the training MS data...")}, () -> {
                    final CmdTask[] commands = new CmdTask[2];
                    try {
                        SwingUtilities.invokeAndWait(() -> {
                            diannReportFileField.setText(diann_report_file);
                            commands[0] = new CmdTask(buildCarafeCommand(), "Carafe", "Run Carafe to generate fine-tuned library ...");
                            commands[1] = new CmdTask(buildDIANNCommand(projectMsFile, carafeLibraryPath, libraryDb, diann_project_dir), "DIA-NN", "DIA-NN search for project data using fine-tuned library ...");
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                        return new CmdTask[0];
                    }
                    return commands;
                });
            }
            default -> JOptionPane.showMessageDialog(this, "Unsupported workflow selected!", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    @FunctionalInterface
    private interface NextCommandsSupplier {
        CmdTask[] getNextCommands();
    }

    private static class CmdTask {
        final String cmd;
        final String tool_name;
        final String task_description;

        CmdTask(String cmd, String tool_name, String task_description) {
            this.cmd = cmd;
            this.tool_name = tool_name;
            this.task_description = task_description;
        }
    }

    private void executeChainedCommands(CmdTask[] initialCommands, NextCommandsSupplier nextCommandsSupplier) {
        isRunning = true;
        runButton.setEnabled(false);
        stopButton.setEnabled(true);
        progressBar.setIndeterminate(true);
        progressBar.setString("Running DIA-NN...");

        Object selectedPython = pythonPathCombo.getSelectedItem();
        String pythonPath = selectedPython != null ? selectedPython.toString().trim() : "";
        if (!pythonPath.isEmpty()) {
            prefs.put(PREF_PYTHON_PATH, pythonPath);
        }

        executor = Executors.newSingleThreadExecutor();
        executor.submit(() -> {
            try {
                for (CmdTask command : initialCommands) {
                    if (!isRunning) return;

                    updateProgressBarForCommand(command.task_description);

                    consoleArea.append("\n========================================\n");
                    consoleArea.append("Running: " + command.task_description + "\n");
                    consoleArea.append("Command: " + command.cmd + "\n");
                    consoleArea.append("========================================\n\n");

                    int exitCode = runSingleCommand(command.cmd, pythonPath);
                    if (exitCode != 0) {
                        SwingUtilities.invokeLater(() -> {
                            consoleArea.append("\n[ERROR] Command failed with exit code: " + exitCode + "\n");
                            progressBar.setString("Failed");
                            finishExecution();
                        });
                        return;
                    }
                }

                if (nextCommandsSupplier != null && isRunning) {
                    CmdTask[] nextCommands = nextCommandsSupplier.getNextCommands();
                    for (CmdTask command : nextCommands) {
                        if (!isRunning) return;

                        updateProgressBarForCommand(command.task_description);

                        consoleArea.append("\n========================================\n");
                        consoleArea.append("Running: " + command.task_description + "\n");
                        consoleArea.append("Command: " + command.cmd + "\n");
                        consoleArea.append("========================================\n\n");

                        int exitCode = runSingleCommand(command.cmd, pythonPath);
                        if (exitCode != 0) {
                            SwingUtilities.invokeLater(() -> {
                                consoleArea.append("\n[ERROR] Command failed with exit code: " + exitCode + "\n");
                                progressBar.setString("Failed");
                                finishExecution();
                            });
                            return;
                        }
                    }
                }

                SwingUtilities.invokeLater(() -> {
                    consoleArea.append("\n[SUCCESS] Workflow completed successfully!\n");
                    progressBar.setString("Completed");
                    finishExecution();
                });

            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    consoleArea.append("\n[ERROR] Error: " + e.getMessage() + "\n");
                    progressBar.setString("Error");
                    finishExecution();
                });
            }
        });
    }

    private void updateProgressBarForCommand(String description) {
        if (progressBar == null) return;
        SwingUtilities.invokeLater(() -> progressBar.setString(description));
    }

    private int runSingleCommand(String command, String pythonPath) throws Exception {
        ProcessBuilder pb = new ProcessBuilder();
        if (System.getProperty("os.name").toLowerCase().contains("windows")) {
            pb.command("cmd", "/c", command);
        } else {
            pb.command("bash", "-c", command);
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

        String lowerCmd = command.toLowerCase();
        if (lowerCmd.contains("diann") && lowerCmd.contains("--f ")) {
            java.util.Map<String, String> env = pb.environment();
            String target_omp_num_threads = "OMP_NUM_THREADS";
            String target_mkl_num_threads = "MKL_NUM_THREADS";
            String target_kmp_affinity = "KMP_AFFINITY";
            try {
                for (String key : env.keySet()) {
                    if (key.equalsIgnoreCase(target_omp_num_threads)) { target_omp_num_threads = key; break; }
                }
                for (String key : env.keySet()) {
                    if (key.equalsIgnoreCase(target_mkl_num_threads)) { target_mkl_num_threads = key; break; }
                }
                for (String key : env.keySet()) {
                    if (key.equalsIgnoreCase(target_kmp_affinity)) { target_kmp_affinity = key; break; }
                }

                String omp = env.get(target_omp_num_threads);
                if (omp == null || omp.trim().isEmpty() || omp.trim().equals("0")) {
                    String threads = String.valueOf(Runtime.getRuntime().availableProcessors());
                    env.put(target_omp_num_threads, threads);
                    env.put(target_mkl_num_threads, threads);
                }
                env.remove(target_kmp_affinity);
                env.put("KMP_WARNINGS", "off");
            } catch (Throwable ignored) {}

            String dbgOmp = pb.environment().getOrDefault(target_omp_num_threads, "(unset)");
            String dbgMkl = pb.environment().getOrDefault(target_mkl_num_threads, "(unset)");
            String dbgKmp = pb.environment().getOrDefault(target_kmp_affinity, "(unset)");
            final String dbgMsg = String.format("[DEBUG] DIANN env: OMP_NUM_THREADS=%s, MKL_NUM_THREADS=%s, KMP_AFFINITY=%s", dbgOmp, dbgMkl, dbgKmp);
            SwingUtilities.invokeLater(() -> {
                consoleArea.append(dbgMsg + "\n");
                consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
            });
        }

        currentProcess = pb.start();

        BufferedReader reader = new BufferedReader(new InputStreamReader(currentProcess.getInputStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            final String output = line;
            SwingUtilities.invokeLater(() -> {
                consoleArea.append(output + "\n");
                consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
            });
        }

        return currentProcess.waitFor();
    }

    private String buildDIANNCommand(String ms_file, String spectral_library_file, String database, String out_dir) {
        Object diannPath = diannPathCombo.getSelectedItem();
        ArrayList<String> diannArgs = new ArrayList<>();
        if (diannPath != null && !diannPath.toString().trim().isEmpty()) {
            String diann_path = "\"" + diannPath.toString().trim() + "\"";
            diannArgs.add(diann_path);

            File F = new File(ms_file);
            int n_ms_files = 0;
            if (F.isFile()) {
                diannArgs.add("--f");
                diannArgs.add("\"" + ms_file + "\"");
                n_ms_files = 1;
            } else if (F.isDirectory()) {
                File analysisTdf = new File(ms_file + File.separator + "analysis.tdf");
                if (analysisTdf.exists()) {
                    diannArgs.add("--f");
                    diannArgs.add("\"" + ms_file + "\"");
                    n_ms_files = 1;
                } else {
                    File[] mzMLFiles = F.listFiles((dir, name) -> name.toLowerCase().endsWith(".mzml"));
                    n_ms_files = 0;
                    if (mzMLFiles != null) {
                        for (File mzMLFile : mzMLFiles) {
                            diannArgs.add("--f");
                            diannArgs.add("\"" + mzMLFile.getPath() + "\"");
                            n_ms_files++;
                        }
                    } else {
                        File[] subDirs = F.listFiles(File::isDirectory);
                        n_ms_files = 0;
                        if (subDirs != null) {
                            for (File subDir : subDirs) {
                                File subAnalysisTdf = new File(subDir.getPath() + File.separator + "analysis.tdf");
                                if (subAnalysisTdf.exists()) {
                                    diannArgs.add("--f");
                                    diannArgs.add("\"" + subDir.getPath() + "\"");
                                    n_ms_files++;
                                }
                            }
                        } else {
                            JOptionPane.showMessageDialog(this, "Please select a valid mzML/timsTOF DIA file or a folder containing mzML files or timsTOF DIA raw files.", "Input Required", JOptionPane.WARNING_MESSAGE);
                            return "";
                        }
                    }
                }
            } else {
                JOptionPane.showMessageDialog(this, "Please select a valid mzML/timsTOF DIA file or a folder containing mzML files or timsTOF DIA raw files.", "Input Required", JOptionPane.WARNING_MESSAGE);
                return "";
            }

            if (spectral_library_file.isEmpty() && !database.isEmpty()) {
                diannArgs.add("--lib");
                diannArgs.add("\"\"");
                diannArgs.add("--gen-spec-lib");
                diannArgs.add("--predictor");
                diannArgs.add("--fasta");
                diannArgs.add("\"" + database + "\"");
                diannArgs.add("--fasta-search");
            } else if (!spectral_library_file.isEmpty() && !database.isEmpty()) {
                diannArgs.add("--lib");
                diannArgs.add("\"" + spectral_library_file + "\"");
                diannArgs.add("--gen-spec-lib");
                diannArgs.add("--reannotate");
                diannArgs.add("--fasta");
                diannArgs.add("\"" + database + "\"");
            } else {
                JOptionPane.showMessageDialog(this, "Please provide a spectral library file or a protein database file.", "Input Required", JOptionPane.WARNING_MESSAGE);
                return "";
            }

            int cores = Runtime.getRuntime().availableProcessors();
            diannArgs.add("--threads");
            diannArgs.add(String.valueOf(cores));
            diannArgs.add("--verbose");
            diannArgs.add("1");

            diannArgs.add("--out");
            diannArgs.add("\"" + out_dir + File.separator + "report.parquet\"");
            diannArgs.add("--out-lib");
            diannArgs.add("\"" + out_dir + File.separator + "report-lib.parquet\"");

            String fixModSelected = fixModSelectedField.getText().trim();
            if (fixModSelected.equalsIgnoreCase("1")) {
                diannArgs.add("--unimod4");
            } else {
                JOptionPane.showMessageDialog(this, "Unsupported modification settings. Please select '1' for Fixed modifications.", "Warning", JOptionPane.WARNING_MESSAGE);
                return "";
            }

            String varModSelected = varModSelectedField.getText().trim();
            if (varModSelected.equalsIgnoreCase("2")) {
                diannArgs.add("--var-mods");
                diannArgs.add(String.valueOf(maxVarSpinner.getValue()));
                diannArgs.add("--var-mod");
                diannArgs.add("UniMod:35,15.994915,M");
            } else if (varModSelected.equalsIgnoreCase("0")) {
                // no modification
            } else {
                JOptionPane.showMessageDialog(this, "Unsupported modification settings. Please select '2' for Variable modifications.", "Warning", JOptionPane.WARNING_MESSAGE);
                return "";
            }

            String enzyme = ((String) enzymeCombo.getSelectedItem()).split(":")[0];
            if (enzyme.equalsIgnoreCase("1")) {
                diannArgs.add("--cut");
                diannArgs.add("\"K*,R*,!*P\"");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else if (enzyme.equalsIgnoreCase("2")) {
                diannArgs.add("--cut");
                diannArgs.add("\"K*,R*\"");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else {
                JOptionPane.showMessageDialog(this, "Unsupported enzyme settings. Please select '1' for trypsin or '2' for chymotrypsin.", "Warning", JOptionPane.WARNING_MESSAGE);
                return "";
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
            diannArgs.add(String.valueOf(minFragMzSpinner.getValue()));
            diannArgs.add("--max-fr-mz");
            diannArgs.add(String.valueOf(maxFragMzSpinner.getValue()));

            diannArgs.add("--qvalue");
            diannArgs.add("0.01");
            diannArgs.add("--matrices");
            if (n_ms_files >= 2) {
                diannArgs.add("--reanalyse");
            }
            diannArgs.add("--rt-profiling");
            diannArgs.add("--export-quant");

            return StringUtils.join(diannArgs, " ");
        } else {
            JOptionPane.showMessageDialog(this, "Please provide a valid DIA-NN executable path.", "Input Required", JOptionPane.WARNING_MESSAGE);
            return "";
        }
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

        if (tabbedPane != null) {
            SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(4));
        }

        Object selectedPython = pythonPathCombo.getSelectedItem();
        String pythonPath = selectedPython != null ? selectedPython.toString().trim() : "";
        if (!pythonPath.isEmpty()) {
            prefs.put(PREF_PYTHON_PATH, pythonPath);
        }

        consoleArea.append("\n========================================\n");
        consoleArea.append("Starting Carafe...\n");
        if (!pythonPath.isEmpty()) {
            consoleArea.append("Python: " + pythonPath + "\n");
        }
        consoleArea.append("Command: " + command.cmd + "\n");
        consoleArea.append("========================================\n\n");

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
                        String pathSeparator = System.getProperty("os.name").toLowerCase().contains("windows") ? ";" : ":";
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
                        consoleArea.append(output + "\n");
                        consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
                    });
                }

                int exitCode = currentProcess.waitFor();
                SwingUtilities.invokeLater(() -> {
                    if (exitCode == 0) {
                        consoleArea.append("\n[SUCCESS] Carafe completed successfully!\n");
                        progressBar.setString("Completed");
                    } else {
                        consoleArea.append("\n[ERROR] Carafe exited with code: " + exitCode + "\n");
                        progressBar.setString("Failed");
                    }
                    finishExecution();
                });

            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    consoleArea.append("\n[ERROR] Error: " + e.getMessage() + "\n");
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
            consoleArea.append("\n[STOPPED] Process stopped by user.\n");
        }
        finishExecution();
    }

    private void finishExecution() {
        isRunning = false;
        runButton.setEnabled(true);
        stopButton.setEnabled(false);
        progressBar.setIndeterminate(false);
        if (executor != null) {
            executor.shutdown();
        }
    }

    private void showHelp() {
        String helpText = """
            Carafe - AI-Powered Spectral Library Generator
            
            Carafe generates experiment-specific in silico spectral libraries 
            using deep learning for DIA data analysis.
            
            Quick Start:
            1. For fine-tuned library generation:
               - Provide PSM file (DIA-NN report.tsv or .parquet)
               - Provide MS file(s) in mzML format
               - Provide protein database (FASTA)
               - Configure settings and click Run
            
            2. For pretrained model library generation:
               - Only provide protein database (FASTA)
               - Set NCE and MS instrument in Advanced settings
               - Click Run
            
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

    public static void main(String[] args) {
        System.setProperty("awt.useSystemAAFontSettings", "on");
        System.setProperty("swing.aatext", "true");

        try {
            boolean dark = prefs.getBoolean(PREF_DARK_MODE, false);
            if (dark) FlatDarkLaf.setup(); else FlatLightLaf.setup();

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
}
