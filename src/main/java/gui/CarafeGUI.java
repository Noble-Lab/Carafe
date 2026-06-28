package main.java.gui;

import java.awt.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.lang.reflect.Array;
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

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import main.java.db.DBGear;
import main.java.input.CModification;
import org.apache.commons.lang3.StringUtils;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;

import com.formdev.flatlaf.FlatDarkLaf;
import com.formdev.flatlaf.FlatLaf;
import com.formdev.flatlaf.FlatLightLaf;

import ai.djl.util.cuda.CudaUtils;
import main.java.input.CParameter;
import main.java.util.Cloger;
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

    /**
     * Global setting: the label color used to flag a parameter whose value differs from its
     * startup default. Change this single line to restyle every "modified" label. Orange was
     * chosen because it stays readable on both the light and dark themes. (Kept non-final so a
     * future GUI control could change it at runtime.)
     */
    private static Color MODIFIED_LABEL_COLOR = new Color(0xE6, 0x7E, 0x22);

    /** Debounce after the last keystroke before re-evaluating a typed field's label color. */
    private static final int SETTING_INDICATOR_DELAY_MS = 2000;

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

    // Search engine selection + Osprey settings. These are field-initialized with defaults
    // so the search-engine choice defaults to DIA-NN (no behavior change) and the Osprey option
    // widgets are never null even before they are added to a settings panel.
    private JComboBox<String> searchEngineCombo = new JComboBox<>(new String[] { "DIA-NN", "Osprey" });
    private JComboBox<String> ospreyPathCombo;
    // Resolution defaults to hram (high-res Orbitrap/Astral, the common Osprey case).
    private JComboBox<String> ospreyResolutionCombo = new JComboBox<>(new String[] { "hram", "unit", "auto" });
    // FDR method is always percolator for Osprey (no UI control; hardcoded in buildOspreyCommand).
    private JComboBox<String> ospreyFdrLevelCombo = new JComboBox<>(new String[] { "precursor", "peptide", "both" });
    private JComboBox<String> ospreySharedPeptidesCombo = new JComboBox<>(new String[] { "all", "razor", "unique" });
    private JTextField ospreyRunFdrField = new JTextField("0.01");
    private JTextField ospreyExperimentFdrField = new JTextField("0.01");
    private JTextField ospreyProteinFdrField = new JTextField("0.01");
    private JTextField ospreyAdditionalOptionsField = new JTextField("");
    // Default off: entrapment (p_target/p_decoy) peptides are only needed for Osprey-side
    // entrapment-FDR validation. Off => Carafe builds a target+decoy library for Osprey.
    private JCheckBox includeEntrapmentCheckbox = new JCheckBox("Include entrapment peptides (Osprey)", false);

    // Initial-library predictor for the Osprey workflows: Carafe's local AlphaPepDeep (default) or
    // a Koina-hosted model. Koina is opt-in and requires network.
    private JComboBox<String> initialLibraryPredictorCombo = new JComboBox<>(new String[] {
            "Carafe (local AlphaPepDeep)",
            "Koina: AlphaPepDeep",
            "Koina: Prosit 2020 HCD",
            "Koina: Prosit 2020 CID",
            "Koina: Prosit timsTOF",
            "Koina: ms2pip HCD",
            "Koina: ms2pip timsTOF"
    });
    private JTextField koinaUrlField = new JTextField("https://koina.wilhelmlab.org");
    // Max number of MSConvert processes to run in parallel when converting raw/.d for Osprey.
    private JTextField ospreyConversionThreadsField = new JTextField("4");

    // Per-step overrides for buildCarafeCommand, used only by the Osprey workflows to reuse
    // the shared Carafe command builder for the peptide-FASTA (NoCut) library and finetune steps.
    // All null for the DIA-NN workflows (1-3), so their behavior is unchanged.
    private String carafeDbOverride = null;         // -db (peptide FASTA instead of libraryDbField)
    private String carafeIOverride = null;          // -i  (Osprey .blib instead of diannReportField)
    private String carafeOutSubdirOverride = null;  // output subdir under the output directory
    private String carafeEnzymeOverride = null;     // -enzyme value (e.g. "NoCut")
    private String carafeSeOverride = null;         // -se value (e.g. "Osprey")
    private String carafeLfTypeOverride = null;     // -lf_type value (e.g. "DIA-NN")

    /** Reset all per-step Carafe command overrides back to default (DIA-NN-workflow) behavior. */
    private void clearCarafeOverrides() {
        carafeDbOverride = null;
        carafeIOverride = null;
        carafeOutSubdirOverride = null;
        carafeEnzymeOverride = null;
        carafeSeOverride = null;
        carafeLfTypeOverride = null;
    }

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
    private java.util.List<JComponent> ospreyExeRowComponents;
    private java.util.List<JComponent> msConvertExeRowComponents;
    private JPanel inputFieldsPanel;

    // Training Data Generation settings
    private JSpinner fdrSpinner;
    private JSpinner ptmSiteProbSpinner;
    private JSpinner ptmSiteQvalueSpinner;
    private JTextField fragTolField;
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
    private JCheckBox benchmarkCheckbox;
    private JLabel benchmarkLabel;

    // Output console
    private JTextArea consoleArea;
    private JProgressBar progressBar;
    private JButton runButton;
    private JButton stopButton;
    private JCheckBox reuseResultsCheckbox;
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
    private static final String PREF_OSPREY_PATH = "ospreyPath";
    private static final String PREF_MSCONVERT_PATH = "msConvertPath";
    private static final String PREF_DARK_MODE = "darkMode";

    /**
     * Time usage tracking map.
     */
    // Synchronized so the parallel Osprey workflow executor can update these from multiple lane
    // threads; the sequential workflows are unaffected.
    private final java.util.Map<String, Double> timeUsageMap =
            java.util.Collections.synchronizedMap(new java.util.LinkedHashMap<>());
    /**
     * List of tasks
     */
    private final java.util.List<CmdTask> tasks =
            java.util.Collections.synchronizedList(new java.util.ArrayList<>());

    /** Live external processes, tracked so Stop can terminate all of them (incl. parallel ones). */
    private final java.util.Set<Process> activeProcesses =
            java.util.Collections.synchronizedSet(new java.util.HashSet<>());

    private String diannVersion = "";
    private boolean isDiannV2 = false;

    private BufferedWriter logWriter;

    public CarafeGUI() {
        setTitle("Carafe - Spectral Library Generator");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Make sure external processes (MSConvert / DIA-NN / Osprey / Carafe / Python) and
        // their children are terminated if the GUI exits without the user clicking Stop (e.g. the
        // window is closed mid-run). Runs on JVM shutdown, which EXIT_ON_CLOSE triggers.
        Runtime.getRuntime().addShutdownHook(new Thread(this::terminateAllProcesses, "carafe-process-cleanup"));
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
        Font defaultFont = UIManager.getFont("Label.font");
        if (defaultFont != null) {
            UIManager.put("defaultFont", defaultFont.deriveFont(13f));
        } else {
            UIManager.put("defaultFont", new Font(Font.SANS_SERIF, Font.PLAIN, 13));
        }
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

    /** Matches ANSI/VT escape sequences (e.g. log4j2 %highlight color codes) so they can be stripped. */
    private static final java.util.regex.Pattern ANSI_ESCAPE = java.util.regex.Pattern.compile("\\e\\[[0-9;]*[A-Za-z]");

    private synchronized void logToConsole(String message) {
        // log4j2's %highlight emits ANSI color codes when stdout is piped (as the GUI captures the
        // DIA-NN/Carafe subprocess output). The JTextArea and log file would otherwise show them as
        // literal characters, so strip them here. Colors are preserved for anyone running the CLI
        // directly in a terminal (that path does not go through this method).
        if (message != null && message.indexOf(27) >= 0) {
            message = ANSI_ESCAPE.matcher(message).replaceAll("");
        }
        final String msg = message;
        SwingUtilities.invokeLater(() -> {
            consoleArea.append(msg);
            consoleArea.setCaretPosition(consoleArea.getDocument().getLength());
        });

        if (logWriter != null) {
            try {
                logWriter.write(msg);
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
        // Uses defaultFont from UIManager if not explicitly set

        inputScrollPane = wrapInScrollPane(createInputPanel());
        tabbedPane.addTab("Workflow", inputScrollPane);
        tabbedPane.addTab("Training Data Generation", wrapInScrollPane(createTrainingDataPanel()));
        tabbedPane.addTab("Model Training", wrapInScrollPane(createModelTrainingPanel()));
        tabbedPane.addTab("Library Generation", wrapInScrollPane(createLibraryGenerationPanel()));
        tabbedPane.addTab("Osprey", wrapInScrollPane(createOspreyPanel()));
        // Console is added last (rightmost); code switches to it via indexOfTab("Console"),
        // so its position is not hardcoded.
        tabbedPane.addTab("Console", createConsolePanel());

        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 15, 10, 15));
        mainPanel.add(tabbedPane, BorderLayout.CENTER);

        add(mainPanel, BorderLayout.CENTER);

        // Footer with run button
        add(createFooter(), BorderLayout.SOUTH);

        // Register the modified-from-default label indicators now that all inputs exist.
        initSettingChangeIndicators();

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
        btn.setFont(btn.getFont().deriveFont(12f));
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
        headerIconLabel.setFont(headerIconLabel.getFont().deriveFont(Font.BOLD, 42f));

        headerTextPanel = new JPanel();
        headerTextPanel.setOpaque(false);
        headerTextPanel.setLayout(new BoxLayout(headerTextPanel, BoxLayout.Y_AXIS));

        headerTitleLabel = new JLabel("Carafe");
        headerTitleLabel.setFont(headerTitleLabel.getFont().deriveFont(Font.BOLD, 28f));

        headerSubtitleLabel = new JLabel("AI-Powered Spectral Library Generator for DIA Proteomics");
        headerSubtitleLabel.setFont(headerSubtitleLabel.getFont().deriveFont(Font.PLAIN, 13f));

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
        headerVersionLabel.setFont(headerVersionLabel.getFont().deriveFont(Font.PLAIN, 12f));
        headerRightPanel.add(headerVersionLabel);

        headerPanel.add(headerTitlePanel, BorderLayout.WEST);
        headerPanel.add(headerRightPanel, BorderLayout.EAST);

        // Ensure buttons panel is on top layer for mouse events when subtitle overlaps
        headerPanel.setComponentZOrder(headerRightPanel, 0);

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
                "3. End-to-end DIA search",
                "4. Osprey: search, finetune, build new library",
                "5. Osprey: end-to-end (finetune, then search project files)"
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

        trainMsRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Train MS File(s):",
                "MS/MS data for model training.\n" +
                        "Supported formats: mzML, Thermo raw, Bruker raw (.d).\n" +
                        "A single MS/MS file, multiple MS/MS files, or a folder containing MS/MS files are accepted.\n"
                        +
                        "Thermo raw files are only supported when starting with DIA-NN search or performing end-to-end DIA search.\n"
                        +
                        "When the format is Thermo raw, MSConvert (ProteoWizard) needs to be installed (convert raw to mzML for Carafe).",
                trainMsFileField = createTextField("Select one or more MS files (or a folder) for training"),
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
                "Protein database or a DIA-NN library used for peptide detection on the train MS file(s).\n" +
                        "Supported formats: FASTA or DIA-NN's speclib (e.g. protein.fasta, protein.fa or *.speclib).",
                trainDbFileField = createTextField("Path to protein FASTA or DIA-NN library for searching the train MS file(s)"),
                createDbButtonsPanel(trainDbFileField, true, "Select a protein database or a DIA-NN library used for peptide detection on the train MS file(s)."));

        // This is the MS/MS data for peptide detection using the fine-tuned spectral
        // library
        projectMsRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Project MS File(s):",
                "MS/MS data for peptide detection using the fine-tuned spectral library using DIA-NN.\n" +
                        "Supported formats: mzML, Thermo raw, Bruker raw (.d).\n" +
                        "A single MS/MS file, multiple MS/MS files, or a folder containing MS/MS files are accepted.\n"
                        +
                        "When the format is Thermo raw, users need to make sure DIA-NN is configured to use Thermo raw format.",
                projectMsFileField = createTextField("Select one or more MS files (or a folder) for analyzing using the fine-tuned spectral library"),
                createMsButtonsPanel(projectMsFileField));

        libraryDbRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Library Protein Database:",
                "Protein database used for fine-tuned spectral library generation.\n" +
                        "Supported formats: FASTA. (e.g. protein.fasta or protein.fa)",
                libraryDbFileField = createTextField("Path to protein FASTA for library generation"),
                createDbButtonsPanel(libraryDbFileField, "Select a protein database for fine-tuned spectral library generation."));

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

        ospreyExeRowComponents = addInputRowToPanel(inputFieldsPanel, gridy++, "Osprey Executable:",
                "Path to the Osprey executable (Osprey.exe on Windows, Osprey on Linux/Mac).\n"
                        + "If left blank, Carafe looks for a bundled build under osprey/<rid>/ next to the jar,\n"
                        + "then ~/.carafe/osprey/<rid>/, then the system PATH.",
                ospreyPathCombo = createOspreyComboBox(),
                createOspreyBrowseButton());

        addInputRowToPanel(inputFieldsPanel, gridy++, "Carafe additional options:",
                "Additional command line options for Carafe.",
                carafeAdditionalOptionsField = createTextField("Carafe additional options"),
                null);

        // Run options + settings load/save, placed just above the Workflow Guide.
        // Use the same row height (COMPONENT_HEIGHT), button style (styleButton) and insets
        // (DEFAULT_INSETS) as the input rows above so this row matches them.
        JPanel workflowOptionsPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 0));
        workflowOptionsPanel.setOpaque(false);

        reuseResultsCheckbox = new JCheckBox("Reuse existing results (skip completed steps)");
        reuseResultsCheckbox.setToolTipText("<html>When enabled, a pipeline step is skipped whenever its expected output file<br>"
                + "already exists in the output directory — letting you resume an interrupted run<br>"
                + "without redoing finished steps such as the DIA-NN search or model training.<br>"
                + "<br>"
                + "Reuse is based only on whether the output file exists; it does not detect<br>"
                + "parameter changes. To force a step to re-run, delete its output first.</html>");
        reuseResultsCheckbox.setPreferredSize(new Dimension(reuseResultsCheckbox.getPreferredSize().width, COMPONENT_HEIGHT));
        workflowOptionsPanel.add(reuseResultsCheckbox);

        // Let the button size naturally to its text (like the Browse/Folder buttons); the
        // checkbox above is pinned to COMPONENT_HEIGHT, so it drives this row's height.
        JButton loadSettingsButton = new JButton("Load Settings");
        styleButton(loadSettingsButton);
        loadSettingsButton.setToolTipText("Load parameter settings from a JSON file");
        loadSettingsButton.addActionListener(e -> loadSettingsDialog());
        workflowOptionsPanel.add(loadSettingsButton);

        GridBagConstraints gbcOptions = new GridBagConstraints();
        gbcOptions.gridx = 0;
        gbcOptions.gridy = gridy++;
        gbcOptions.gridwidth = 3;
        gbcOptions.fill = GridBagConstraints.HORIZONTAL;
        gbcOptions.insets = DEFAULT_INSETS;
        inputFieldsPanel.add(workflowOptionsPanel, gbcOptions);

        JPanel infoWrapper = new JPanel(new BorderLayout());
        infoWrapper.add(createInfoCard(
                "Workflow Guide",
                "Workflow 1: Generate spectral library by running DIA-NN search first\n" +
                        "  - Requires: Train MS files, Train database, Library database\n\n" +
                        "Workflow 2: Generate spectral library from existing DIA-NN results\n" +
                        "  - Requires: DIA-NN report file, Train MS files, Library database\n\n" +
                        "Workflow 3: Complete DIA analysis pipeline (Carafe+DIA-NN)\n" +
                        "  - Requires: Train MS, Project MS, both databases\n\n" +
                        "Workflow 4: Osprey search, finetune, build new library\n" +
                        "  - Requires: Train MS, Train database, Library database\n\n" +
                        "Workflow 5: Osprey end-to-end (finetune, then search project files)\n" +
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
        JPanel msButtonsPanel = new JPanel(new GridLayout(1, 0, 5, 0));

        JButton browse;
        final java.util.List<String> associatedList;

        if (targetField == trainMsFileField) {
            associatedList = trainMsFiles;
            browse = createMultiFileBrowseButton(targetField, "mzML/raw Files", new String[] { "mzML", "raw" },
                    associatedList,"Select one or more mzML/raw files or timsTOF .d folders");
            setupMultiFileFieldInteraction(targetField, associatedList);
        } else if (targetField == projectMsFileField) {
            associatedList = projectMsFiles;
            browse = createMultiFileBrowseButton(targetField, "mzML/raw Files", new String[] { "mzML", "raw" },
                    associatedList,"Select one or more mzML/raw files or timsTOF .d folders");
            setupMultiFileFieldInteraction(targetField, associatedList);
        } else {
            associatedList = null;
            browse = createBrowseButton(targetField, "mzML/raw Files", new String[] { "mzML", "raw" });
        }

        msButtonsPanel.add(browse);

        // Custom Folder Button logic to ensure list is cleared
        JButton folderBtn = new JButton("Folder");
        styleButton(folderBtn);
        folderBtn.setToolTipText("Select a folder containing mzML/raw files or timsTOF .d folders");
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
            java.util.List<String> fileList, String toolTipText) {
        JButton button = new JButton("Browse");
        styleButton(button);
        if(toolTipText != null && !toolTipText.isEmpty()){
            button.setToolTipText(toolTipText);
        }
        button.addActionListener(e -> {
            chooseFiles("Select Files", extensions, description, files -> {
                // Validation: Check for mixed types (mzML, raw, TIMSTOF .d folders)
                boolean hasMzML = false;
                boolean hasRaw = false;
                boolean hasTimsTof = false;
                java.util.List<String> invalidDFolders = new java.util.ArrayList<>();
                boolean hasNonDFolder = false;

                for (File f : files) {
                    String name = f.getName().toLowerCase();
                    if (f.isDirectory()) {
                        if (name.endsWith(".d")) {
                            // Validate TIMSTOF folder contains analysis.tdf
                            File analysisTdf = new File(f, "analysis.tdf");
                            if (analysisTdf.exists()) {
                                hasTimsTof = true;
                            } else {
                                invalidDFolders.add(f.getName());
                            }
                        } else {
                            // Regular folder (not a .d folder)
                            hasNonDFolder = true;
                        }
                    } else if (f.isFile()) {
                        if (name.endsWith(".mzml"))
                            hasMzML = true;
                        if (name.endsWith(".raw"))
                            hasRaw = true;
                    }
                }

                // Check if user selected a single regular folder (not a .d folder)
                if (files.length == 1 && hasNonDFolder) {
                    JOptionPane.showMessageDialog(this,
                            "The selected folder is not a timsTOF .d folder.\n" +
                                    "If you want to select a folder which contains the training MS files, " +
                                    "please use the Folder button.",
                            "Invalid Selection",
                            JOptionPane.WARNING_MESSAGE);
                    fileList.clear();
                    updateFileFieldState(targetField, fileList);
                    targetField.setText("");
                    return;
                }

                // Show error for invalid .d folders (missing analysis.tdf)
                if (!invalidDFolders.isEmpty()) {
                    JOptionPane.showMessageDialog(this,
                            "The following .d folders do not contain analysis.tdf:\n" +
                                    String.join(", ", invalidDFolders),
                            "Invalid TIMSTOF Folder",
                            JOptionPane.WARNING_MESSAGE);
                    fileList.clear();
                    updateFileFieldState(targetField, fileList);
                    targetField.setText("");
                    return;
                }

                // Prevent mixing different data types
                int typeCount = (hasMzML ? 1 : 0) + (hasRaw ? 1 : 0) + (hasTimsTof ? 1 : 0);
                if (typeCount > 1) {
                    JOptionPane.showMessageDialog(this,
                            "Please select only one type: mzML files, RAW files, or TIMSTOF .d folders.",
                            "Invalid Selection",
                            JOptionPane.WARNING_MESSAGE);
                    return;
                }

                fileList.clear();
                for (File f : files) {
                    // Only add valid files/folders
                    if (f.isDirectory() && f.getName().toLowerCase().endsWith(".d")) {
                        fileList.add(f.getAbsolutePath());
                    } else if (f.isFile()) {
                        fileList.add(f.getAbsolutePath());
                    }
                }
                updateFileFieldState(targetField, fileList);
                if (files.length > 0) {
                    File parent = files[0].getParentFile();
                    if (parent != null) {
                        prefs.put(PREF_LAST_DIR, parent.getAbsolutePath());
                    }
                }
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
            // If all files were removed, clear the text field explicitly
            if (newPaths.isEmpty()) {
                field.setText("");
            }
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
                setVisible(ospreyExeRowComponents, false);
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
                setVisible(ospreyExeRowComponents, false);
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
                setVisible(ospreyExeRowComponents, false);
                setVisible(diannAdditionalOptionsRowComponents, true);
                updateMsConvertVisibility();
            }
            case 3 -> { // Workflow 4: Osprey search -> finetune -> new library
                setVisible(diannReportRowComponents, false);
                setVisible(trainMsRowComponents, true);
                setVisible(trainDbRowComponents, true);
                setVisible(projectMsRowComponents, false);
                setVisible(libraryDbRowComponents, true);
                setVisible(diannExeRowComponents, false);
                setVisible(ospreyExeRowComponents, true);
                setVisible(diannAdditionalOptionsRowComponents, false);
                updateMsConvertVisibility();
            }
            case 4 -> { // Workflow 5: Osprey end-to-end (then search project files)
                setVisible(diannReportRowComponents, false);
                setVisible(trainMsRowComponents, true);
                setVisible(trainDbRowComponents, true);
                setVisible(projectMsRowComponents, true);
                setVisible(libraryDbRowComponents, true);
                setVisible(diannExeRowComponents, false);
                setVisible(ospreyExeRowComponents, true);
                setVisible(diannAdditionalOptionsRowComponents, false);
                updateMsConvertVisibility();
            }
        }

        // Show benchmark checkbox only for Workflow 3 (End-to-end)
        boolean showBenchmark = (globalWorkflowIndex == 2);
        if (benchmarkCheckbox != null) {
            benchmarkCheckbox.setVisible(showBenchmark);
        }
        if (benchmarkLabel != null) {
            benchmarkLabel.setVisible(showBenchmark);
        }

        stripScrollPaneBorder(inputFieldsPanel);
        inputFieldsPanel.setBorder(BorderFactory.createEmptyBorder());
        inputFieldsPanel.revalidate();
        inputFieldsPanel.repaint();
    }

    /** Settings panel for the Osprey search engine (Workflows 4 and 5). */
    private JPanel createOspreyPanel() {
        JPanel panel = new ScrollablePanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = DEFAULT_INSETS;
        gbc.anchor = GridBagConstraints.WEST;
        int row = 0;

        styleComboBox(ospreyResolutionCombo);
        styleComboBox(ospreyFdrLevelCombo);
        styleComboBox(ospreySharedPeptidesCombo);
        styleComboBox(initialLibraryPredictorCombo);

        // Initial-library predictor (Carafe local vs Koina) + Koina server URL.
        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0;
        panel.add(createLabel("Initial library predictor:",
                "How the initial (non-finetuned) library is predicted before the first Osprey search.\n"
                        + "Carafe local runs AlphaPepDeep on your machine; Koina options query the Koina web\n"
                        + "service (needs network). The finetuned library is always built locally by Carafe."), gbc);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(initialLibraryPredictorCombo, gbc);
        row++;

        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0;
        panel.add(createLabel("Koina server URL:",
                "Koina server base URL (used only when a Koina predictor is selected)."), gbc);
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(koinaUrlField, gbc);
        row++;

        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.weightx = 0;
        panel.add(includeEntrapmentCheckbox, gbc);
        includeEntrapmentCheckbox.setToolTipText(
                "Include FDRBench entrapment (p_target/p_decoy) peptides in the target-decoy library.\n"
                        + "Off by default; enable for Osprey-side entrapment-FDR validation.");
        row++;

        Object[][] combos = {
                { "Resolution:", ospreyResolutionCombo,
                        "Spectral resolution mode passed to Osprey (--resolution). Default: hram." },
                { "FDR level:", ospreyFdrLevelCombo, "Osprey --fdr-level." },
                { "Shared peptides:", ospreySharedPeptidesCombo, "Osprey --shared-peptides." },
        };
        for (Object[] c : combos) {
            gbc.gridx = 0;
            gbc.gridy = row;
            gbc.weightx = 0;
            panel.add(createLabel((String) c[0], (String) c[2]), gbc);
            gbc.gridx = 1;
            gbc.weightx = 1;
            panel.add((JComponent) c[1], gbc);
            row++;
        }

        Object[][] fields = {
                { "Run FDR:", ospreyRunFdrField, "Per-run FDR threshold (--run-fdr)." },
                { "Experiment FDR:", ospreyExperimentFdrField, "Experiment-wide FDR threshold (--experiment-fdr)." },
                { "Protein FDR:", ospreyProteinFdrField, "Protein-level FDR threshold (--protein-fdr)." },
                { "Additional options:", ospreyAdditionalOptionsField,
                        "Extra command-line options appended to the Osprey command." },
                { "Parallel MSConvert processes:", ospreyConversionThreadsField,
                        "How many raw/.d files to convert to mzML at once (separate MSConvert processes).\n"
                                + "Used by Workflows 4/5 when converting acquisition files for Osprey." },
        };
        for (Object[] f : fields) {
            gbc.gridx = 0;
            gbc.gridy = row;
            gbc.weightx = 0;
            panel.add(createLabel((String) f[0], (String) f[2]), gbc);
            gbc.gridx = 1;
            gbc.weightx = 1;
            panel.add((JComponent) f[1], gbc);
            row++;
        }

        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.gridwidth = 2;
        gbc.weightx = 1;
        panel.add(createInfoCard("Osprey",
                "Osprey is used by Workflows 4 and 5. Carafe builds a target-decoy library\n"
                        + "(with the digest options from the Library Generation tab), runs Osprey,\n"
                        + "finetunes its models on the resulting .blib, then predicts a new library.\n"
                        + "Fragment tolerance is taken from the Training Data Generation tab."), gbc);

        return panel;
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

        fragTolField = createTextField("20");
        fragTolField.setText("20");
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(fragTolField, gbc);

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

        xicCorSpinner = createDoubleSpinner(0.8, 0.0, 1.0, 0.01);
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
                        "When it is set to 'auto', Carafe determines the NCE from the MS/MS data and uses it\n" +
                        "(this also applies to the Osprey initial library, including Koina predictions).\n" +
                        "Enter a number to override with a specific NCE."),
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

        if(DBGear.get_enzymes().isEmpty()){
            DBGear.init_enzymes();
        }

        String[] enzymes = new String[DBGear.get_enzymes().size()];
        for(int i = 0; i < DBGear.get_enzymes().size(); i++){
            enzymes[i] = i + ": " + DBGear.get_enzymes().get(i).getName();
        }

        // String[] enzymes = {
        //         "1: Trypsin (default)",
        //         "2: Trypsin (no P rule)",
        //         "3: Arg-C",
        //         "4: Arg-C (no P rule)",
        //         "5: Arg-N",
        //         "6: Glu-C",
        //         "7: Lys-C",
        //         "0: Non enzyme"
        // };
        enzymeCombo = new JComboBox<>(enzymes);
        styleComboBox(enzymeCombo);
        enzymeCombo.setSelectedIndex(DBGear.getEnzymeIndexByName("Trypsin (no P rule)"));
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
                    String newMod = parts[0].trim();
                    // If selecting "0" (no modification), replace everything with "0"
                    if (newMod.equals("0")) {
                        fixModSelectedField.setText("0");
                        return;
                    }
                    String currentValue = fixModSelectedField.getText().trim();
                    if (currentValue.isEmpty() || currentValue.equals("0")) {
                        // Replace "0" or empty with new selection
                        fixModSelectedField.setText(newMod);
                    } else {
                        // Append if not already present (check each mod ID)
                        java.util.Set<String> existingMods = new java.util.LinkedHashSet<>();
                        for (String mod : currentValue.split(",")) {
                            String trimmed = mod.trim();
                            if (!trimmed.isEmpty())
                                existingMods.add(trimmed);
                        }
                        for (String mod : newMod.split(",")) {
                            String trimmed = mod.trim();
                            if (!trimmed.isEmpty())
                                existingMods.add(trimmed);
                        }
                        // Remove "0" if present since we're adding actual mods
                        existingMods.remove("0");
                        fixModSelectedField.setText(String.join(",", existingMods));
                    }
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
        // Auto-cleanup: remove "0" when other mods are present
        fixModSelectedField.getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
            private boolean updating = false;

            private void cleanup() {
                if (updating)
                    return;
                String text = fixModSelectedField.getText().trim();
                if (text.contains(",")) {
                    java.util.List<String> mods = new java.util.ArrayList<>();
                    boolean hasZero = false;
                    for (String m : text.split(",")) {
                        String trimmed = m.trim();
                        if (trimmed.equals("0")) {
                            hasZero = true;
                        } else if (!trimmed.isEmpty()) {
                            mods.add(trimmed);
                        }
                    }
                    // Only cleanup if there was a standalone "0" with other mods
                    if (hasZero && !mods.isEmpty()) {
                        String cleaned = String.join(",", mods);
                        if (!cleaned.equals(text)) {
                            updating = true;
                            SwingUtilities.invokeLater(() -> {
                                fixModSelectedField.setText(cleaned);
                                updating = false;
                            });
                        }
                    }
                }
            }

            public void insertUpdate(javax.swing.event.DocumentEvent e) {
                cleanup();
            }

            public void removeUpdate(javax.swing.event.DocumentEvent e) {
                cleanup();
            }

            public void changedUpdate(javax.swing.event.DocumentEvent e) {
                cleanup();
            }
        });
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
                    String newMod = parts[0].trim();
                    // If selecting "0" (no modification), replace everything with "0"
                    if (newMod.equals("0")) {
                        varModSelectedField.setText("0");
                        return;
                    }
                    String currentValue = varModSelectedField.getText().trim();
                    if (currentValue.isEmpty() || currentValue.equals("0")) {
                        // Replace "0" or empty with new selection
                        varModSelectedField.setText(newMod);
                    } else {
                        // Append if not already present (check each mod ID)
                        java.util.Set<String> existingMods = new java.util.LinkedHashSet<>();
                        for (String mod : currentValue.split(",")) {
                            String trimmed = mod.trim();
                            if (!trimmed.isEmpty())
                                existingMods.add(trimmed);
                        }
                        for (String mod : newMod.split(",")) {
                            String trimmed = mod.trim();
                            if (!trimmed.isEmpty())
                                existingMods.add(trimmed);
                        }
                        // Remove "0" if present since we're adding actual mods
                        existingMods.remove("0");
                        varModSelectedField.setText(String.join(",", existingMods));
                    }
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
        // Auto-cleanup: remove "0" when other mods are present
        varModSelectedField.getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
            private boolean updating = false;

            private void cleanup() {
                if (updating)
                    return;
                String text = varModSelectedField.getText().trim();
                if (text.contains(",")) {
                    java.util.List<String> mods = new java.util.ArrayList<>();
                    boolean hasZero = false;
                    for (String m : text.split(",")) {
                        String trimmed = m.trim();
                        if (trimmed.equals("0")) {
                            hasZero = true;
                        } else if (!trimmed.isEmpty()) {
                            mods.add(trimmed);
                        }
                    }
                    // Only cleanup if there was a standalone "0" with other mods
                    if (hasZero && !mods.isEmpty()) {
                        String cleaned = String.join(",", mods);
                        if (!cleaned.equals(text)) {
                            updating = true;
                            SwingUtilities.invokeLater(() -> {
                                varModSelectedField.setText(cleaned);
                                updating = false;
                            });
                        }
                    }
                }
            }

            public void insertUpdate(javax.swing.event.DocumentEvent e) {
                cleanup();
            }

            public void removeUpdate(javax.swing.event.DocumentEvent e) {
                cleanup();
            }

            public void changedUpdate(javax.swing.event.DocumentEvent e) {
                cleanup();
            }
        });
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

        minPepMzSpinner = createSpinner(400, 100, 2000, 50);
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

        maxPepMzSpinner = createSpinner(900, 100, 3000, 50);
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

        maxPepChargeSpinner = createSpinner(3, 1, 10, 1);
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

        libMaxFragMzSpinner = createSpinner(1960, 500, 3000, 1);
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
        gbc.weightx = 0;
        gbc.gridwidth = 1;
        benchmarkLabel = createLabel("Benchmark (DIA-NN library-free):",
                "When enabled, runs an additional DIA-NN library-free search on project files for comparison.\n" +
                        "This option only applies to Workflow 3 (End-to-end DIA search).");
        benchmarkLabel.setVisible(false); // Hidden by default, shown only for Workflow 3
        panel.add(benchmarkLabel, gbc);

        benchmarkCheckbox = new JCheckBox();
        benchmarkCheckbox.setSelected(false);
        benchmarkCheckbox.setVisible(false); // Hidden by default, shown only for Workflow 3
        gbc.gridx = 1;
        gbc.weightx = 1;
        panel.add(benchmarkCheckbox, gbc);

        gbc.gridx = 0;
        gbc.gridy = 21;
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
        // label.setFont(...) will inherit from UIManager defaultFont (13pt)
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

    private JPanel createDbButtonsPanel(JTextField targetField, String tipString) {
        JPanel panel = new JPanel(new GridLayout(1, 0, 5, 0));
        panel.setOpaque(false);

        JButton browseButton = new JButton("Browse");
        styleButton(browseButton);
        if(tipString != null && !tipString.isEmpty()){
            browseButton.setToolTipText(tipString);
        }
        browseButton.addActionListener(e -> {
            chooseFile("FASTA File (*.fasta, *.fa)", JFileChooser.FILES_ONLY,
                    new FileNameExtensionFilter("FASTA File (*.fasta, *.fa)", "fasta", "fa"), f -> {
                        targetField.setText(f.getAbsolutePath());
                        prefs.put(PREF_LAST_DIR, f.getParent());
                    });
        });

        JButton downloadButton = new JButton("Download");
        styleButton(downloadButton);
        downloadButton.setToolTipText("Download protein database from UniProt");
        downloadButton.addActionListener(e -> {
            UniProtDownloadDialog dialog = new UniProtDownloadDialog(this, targetField, outputDirField);
            dialog.showDialog();
        });

        panel.add(browseButton);
        panel.add(downloadButton);
        return panel;
    }

    /**
     * Create buttons for database selection, supporting both FASTA files and spectral libraries.
     * This is current used for train protein database input.
     * @param targetField The text field to store the selected file path.
     * @param support_speclib Whether to support spectral library files.
     * @return A panel containing the browse and download buttons.
     */
    private JPanel createDbButtonsPanel(JTextField targetField, boolean support_speclib, String tipString) {
        JPanel panel = new JPanel(new GridLayout(1, 0, 5, 0));
        panel.setOpaque(false);

        JButton browseButton = new JButton("Browse");
        styleButton(browseButton);
        if(tipString != null && !tipString.isEmpty()){
            browseButton.setToolTipText(tipString);
        }
        if(support_speclib){
            browseButton.addActionListener(e -> {
                chooseFile("FASTA File (*.fasta, *.fa) or Spectral Library File (*.speclib)", JFileChooser.FILES_ONLY,
                        new FileNameExtensionFilter("FASTA File (*.fasta, *.fa) or Spectral Library File (*.speclib)", "fasta", "fa", "speclib"), f -> {
                            targetField.setText(f.getAbsolutePath());
                            prefs.put(PREF_LAST_DIR, f.getParent());
                        });
            });
        }else{
            browseButton.addActionListener(e -> {
                chooseFile("FASTA File (*.fasta, *.fa)", JFileChooser.FILES_ONLY,
                        new FileNameExtensionFilter("FASTA File (*.fasta, *.fa)", "fasta", "fa"), f -> {
                            targetField.setText(f.getAbsolutePath());
                            prefs.put(PREF_LAST_DIR, f.getParent());
                        });
            });
        }

        JButton downloadButton = new JButton("Download");
        styleButton(downloadButton);
        downloadButton.setToolTipText("Download protein database from UniProt");
        downloadButton.addActionListener(e -> {
            UniProtDownloadDialog dialog = new UniProtDownloadDialog(this, targetField, outputDirField);
            dialog.showDialog();
        });

        panel.add(browseButton);
        panel.add(downloadButton);
        return panel;
    }

    private JPanel createFolderButton(JTextField targetField) {
        JPanel panel = new JPanel(new GridLayout(1, 0, 5, 0));
        panel.setOpaque(false);

        JButton folderButton = new JButton("Folder");
        styleButton(folderButton);
        folderButton.addActionListener(e -> {
            chooseFile("Select Folder", JFileChooser.DIRECTORIES_ONLY, null, f -> {
                targetField.setText(f.getAbsolutePath());
                prefs.put(PREF_LAST_DIR, f.getAbsolutePath());
            });
        });

        JButton openButton = new JButton("Open");
        styleButton(openButton);
        openButton.putClientProperty("carafe.noFreeze", true);
        openButton.setToolTipText("Open the output directory in file explorer");
        openButton.addActionListener(e -> {
            String path = targetField.getText().trim();
            if (!path.isEmpty()) {
                File dir = new File(path);
                if (dir.exists() && dir.isDirectory()) {
                    try {
                        java.awt.Desktop.getDesktop().open(dir);
                    } catch (IOException ex) {
                        JOptionPane.showMessageDialog(this,
                                "Failed to open directory: " + ex.getMessage(),
                                "Error", JOptionPane.ERROR_MESSAGE);
                    }
                } else {
                    JOptionPane.showMessageDialog(this,
                            "Directory does not exist: " + path,
                            "Directory Not Found", JOptionPane.WARNING_MESSAGE);
                }
            }
        });

        panel.add(folderButton);
        panel.add(openButton);
        return panel;
    }

    private JPanel createPythonBrowseButton() {
        JPanel panel = new JPanel(new GridLayout(1, 0, 5, 0));
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
        install.setToolTipText("Install Carafe dependent Python and Python packages automatically.");
        install.addActionListener(e -> {
            install.setEnabled(false);
            browse.setEnabled(false);
            CarafeGUI.this.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));

            new Thread(() -> {
                String home = System.getProperty("user.home");
                java.nio.file.Path installRoot = Paths.get(home, ".carafe");
                try {
                    SwingUtilities.invokeLater(() -> {
                        if (tabbedPane != null) {
                            int consoleIdx = tabbedPane.indexOfTab("Console");
                            tabbedPane.setSelectedIndex(consoleIdx >= 0 ? consoleIdx
                                    : Math.max(0, tabbedPane.getTabCount() - 1));
                        }
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

                            // Auto-correct: if user selected DIA-NN.exe, check for diann.exe
                            if (System.getProperty("os.name").toLowerCase().contains("windows")
                                    && selectedFile.getName().equalsIgnoreCase("DIA-NN.exe")) {
                                File diannExe = new File(selectedFile.getParent(), "diann.exe");
                                if (diannExe.exists()) {
                                    path = diannExe.getAbsolutePath();
                                    JOptionPane.showMessageDialog(this,
                                            "Auto-corrected to diann.exe (command-line version).\n" +
                                                    "DIA-NN.exe is the GUI, diann.exe is required for batch processing.",
                                            "Path Corrected", JOptionPane.INFORMATION_MESSAGE);
                                }
                            }

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

    private JComboBox<String> createOspreyComboBox() {
        JComboBox<String> combo = new JComboBox<>();
        combo.setEditable(true);
        combo.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        combo.setToolTipText("Select the Osprey executable, or leave blank to use a bundled/auto-detected build");

        String ospreyPrototype = System.getProperty("os.name").toLowerCase().contains("windows")
                ? "C:\\Carafe\\osprey\\win-x64\\Osprey.exe"
                : "/usr/local/bin/Osprey";
        combo.setPrototypeDisplayValue(ospreyPrototype);

        String saved = prefs.get(PREF_OSPREY_PATH, "");
        if (!saved.isEmpty()) {
            combo.addItem(saved);
        }
        // Prefill with an auto-resolved (bundled/PATH) executable if one is found.
        String resolved = resolveOspreyBinary();
        if (!resolved.isEmpty()) {
            boolean present = false;
            for (int i = 0; i < combo.getItemCount(); i++) {
                if (combo.getItemAt(i).equals(resolved)) {
                    present = true;
                    break;
                }
            }
            if (!present) {
                combo.addItem(resolved);
            }
        }
        if (!saved.isEmpty()) {
            combo.setSelectedItem(saved);
        } else if (combo.getItemCount() > 0) {
            combo.setSelectedIndex(0);
        }

        combo.addActionListener(e -> {
            Object selected = combo.getSelectedItem();
            if (selected != null) {
                prefs.put(PREF_OSPREY_PATH, selected.toString());
            }
        });
        return combo;
    }

    private JButton createOspreyBrowseButton() {
        JButton button = new JButton("Browse");
        styleButton(button);
        button.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser(prefs.get(PREF_LAST_DIR, System.getProperty("user.home")));
            chooser.setDialogTitle("Select the Osprey executable");
            chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                File selectedFile = chooser.getSelectedFile();
                String path = selectedFile.getAbsolutePath();
                boolean found = false;
                for (int i = 0; i < ospreyPathCombo.getItemCount(); i++) {
                    if (ospreyPathCombo.getItemAt(i).equals(path)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    ospreyPathCombo.addItem(path);
                }
                ospreyPathCombo.setSelectedItem(path);
                prefs.put(PREF_OSPREY_PATH, path);
                if (selectedFile.getParent() != null) {
                    prefs.put(PREF_LAST_DIR, selectedFile.getParent());
                }
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

        // Osprey reads only mzML, so its workflows also need MSConvert for Bruker .d inputs
        // (DIA-NN workflows read .d directly and keep the .raw-only rule above).
        if (!show && (globalWorkflowIndex == 3 || globalWorkflowIndex == 4)) {
            show = ospreyNeedsConversion(trainMsFiles, trainMsFileField)
                    || ospreyNeedsConversion(projectMsFiles, projectMsFileField);
        }

        setVisible(msConvertExeRowComponents, show);
        if (inputFieldsPanel != null) {
            inputFieldsPanel.revalidate();
            inputFieldsPanel.repaint();
        }
    }

    /** True if the selection contains any non-mzML acquisition file (.raw or Bruker .d) that
     *  Osprey would require MSConvert to convert. */
    private boolean ospreyNeedsConversion(java.util.List<String> selected, JTextField field) {
        java.util.List<String> cands = new java.util.ArrayList<>();
        if (selected != null && !selected.isEmpty()) {
            cands.addAll(selected);
        } else if (field != null && !field.getText().trim().isEmpty()) {
            cands.add(field.getText().trim());
        }
        for (String p : cands) {
            String low = p.toLowerCase();
            File f = new File(p);
            if (low.endsWith(".raw") || (low.endsWith(".d") && f.isDirectory())) {
                return true;
            }
            if (f.isDirectory()) {
                File[] hits = f.listFiles((d, n) -> {
                    String x = n.toLowerCase();
                    return x.endsWith(".raw") || x.endsWith(".d");
                });
                if (hits != null && hits.length > 0) {
                    return true;
                }
            }
        }
        return false;
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
        button.setFont(button.getFont().deriveFont(Font.BOLD, 14f));
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
        button.setFont(button.getFont().deriveFont(Font.PLAIN, 13f));
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
        titleLabel.setFont(titleLabel.getFont().deriveFont(Font.BOLD, 13f));
        titleLabel.setBorder(BorderFactory.createEmptyBorder(0, 0, 6, 0));
        card.add(titleLabel, BorderLayout.NORTH);

        JTextArea contentArea = new JTextArea(content) {
            @Override
            public Dimension getPreferredSize() {
                Dimension d = super.getPreferredSize();
                return new Dimension(200, d.height);
            }
        };
        contentArea.setFont(contentArea.getFont().deriveFont(Font.PLAIN, 12f));
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
        return buildCarafeCommand(null, false);
    }

    private CmdTask buildCarafeCommand(String trainMsFileOverride, boolean isTimsTOF) {
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

        Object selectedPython = pythonPathCombo != null ? pythonPathCombo.getSelectedItem() : null;
        String pythonPath = selectedPython != null ? selectedPython.toString().trim() : "";
        if (!pythonPath.isEmpty()) {
            commandArgs.add("-python");
            commandArgs.add(pythonPath);
        }

        String libraryDb = carafeDbOverride != null ? carafeDbOverride : libraryDbFileField.getText().trim();
        if (!libraryDb.isEmpty()) {
            // cmd.append("-db \"").append(libraryDb).append("\" ");
            commandArgs.add("-db");
            commandArgs.add(libraryDb);
            // commandArgs.add("\"" + libraryDb + "\"");
        }

        String diannReport = carafeIOverride != null ? carafeIOverride : diannReportFileField.getText().trim();
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

        String outSubdir = carafeOutSubdirOverride != null ? carafeOutSubdirOverride : "carafe_library";
        String outDir = outputDirField.getText().trim();
        if (!outDir.isEmpty()) {
            carafe_library_directory = outDir + File.separator + outSubdir;
            // cmd.append("-o \"").append(carafe_library_directory).append("\" ");
            commandArgs.add("-o");
            commandArgs.add(carafe_library_directory);
            // commandArgs.add("\"" + carafe_library_directory + "\"");
        }else{
            carafe_library_directory = outSubdir;
            commandArgs.add("-o");
            commandArgs.add(carafe_library_directory);
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
        commandArgs.add(fragTolField.getText().trim());
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

        // Add -ccs flag for TIMSTOF data
        if (isTimsTOF) {
            commandArgs.add("-ccs");
        }

        // For the Osprey peptide-FASTA path, override the enzyme with "NoCut" so AlphaPepDeep
        // predicts each pre-digested peptide (target and decoy) as-is.
        String enzyme = carafeEnzymeOverride != null
                ? carafeEnzymeOverride
                : ((String) enzymeCombo.getSelectedItem()).split(":")[0];
        // cmd.append("-enzyme ").append(enzyme).append(" ");
        commandArgs.add("-enzyme");
        commandArgs.add(enzyme);
        // cmd.append("-miss_c ").append(missCleavageSpinner.getValue()).append(" ");
        commandArgs.add("-miss_c");
        commandArgs.add(missCleavageSpinner.getValue().toString());

        String fixModSelected = fixModSelectedField.getText().trim();
        if (!fixModSelected.isEmpty()) {
            // cmd.append("-fixMod ").append(fixModSelected).append(" ");
            // remove "," at the start and end
            fixModSelected = fixModSelected.replaceAll("^,", "");
            fixModSelected = fixModSelected.replaceAll(",$", "");
            commandArgs.add("-fixMod");
            commandArgs.add(fixModSelected);
        }

        String varModSelected = varModSelectedField.getText().trim();
        if (!varModSelected.isEmpty()) {
            // cmd.append("-varMod ").append(varModSelected).append(" ");
            // remove "," at the start and end
            varModSelected = varModSelected.replaceAll("^,", "");
            varModSelected = varModSelected.replaceAll(",$", "");
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
        commandArgs.add(carafeLfTypeOverride != null
                ? carafeLfTypeOverride
                : libraryFormatCombo.getSelectedItem().toString());
        // cmd.append("-se DIA-NN ");

        commandArgs.add("-se");
        commandArgs.add(carafeSeOverride != null ? carafeSeOverride : getSelectedSearchEngine());

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
            int pythonIndex = additionalOptionList.indexOf("-python");
            if (pythonIndex >= 0) {
                JOptionPane.showMessageDialog(this, "Please use the Python Executable field instead of adding -python manually.",
                        "Carafe setting", JOptionPane.ERROR_MESSAGE);
                return null;
            }
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
        cmdTask.out_dir = carafe_library_directory;
        cmdTask.out_files.add(carafe_library_directory+File.separator+"carafe_spectral_library.tsv");
        cmdTask.out_files_description.add("Carafe fine-tuned spectral library");
        // Inputs read by this step: protein DB (-db), training MS (-ms), and peptide IDs (-i).
        if (!libraryDb.isEmpty()) {
            cmdTask.input_files.add(libraryDb);
        }
        if (!trainMsFile.isEmpty()) {
            cmdTask.input_files.add(trainMsFile);
        }
        if (!diannReport.isEmpty()) {
            cmdTask.input_files.add(diannReport);
        }

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
                boolean isTimsTOF = false;

                // Resolve inputs
                if (!trainMsFiles.isEmpty()) {
                    for (String path : trainMsFiles) {
                        File pathFile = new File(path);
                        if (path.toLowerCase().endsWith(".d") && pathFile.isDirectory()) {
                            // TIMSTOF .d folder
                            File analysisTdf = new File(pathFile, "analysis.tdf");
                            if (analysisTdf.exists()) {
                                finalMsFiles.add(path);
                                isTimsTOF = true;
                            }
                        } else if (path.toLowerCase().endsWith(".raw")) {
                            rawFilesToConvert.add(path);
                        } else {
                            finalMsFiles.add(path);
                        }
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
                    conversionTask.out_dir = subDir;
                    conversionTask.input_files.addAll(rawFilesToConvert);

                    for (String rawPath : rawFilesToConvert) {
                        String rawName = new File(rawPath).getName();
                        String baseName = rawName.lastIndexOf('.') > 0
                                ? rawName.substring(0, rawName.lastIndexOf('.'))
                                : rawName;
                        String mzML = subDir + File.separator + baseName + ".mzML";
                        finalMsFiles.add(mzML);
                        if (conversionTask.skip_check_file == null) {
                            conversionTask.skip_check_file = mzML;
                        }
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
                    diannTask.out_dir = diann_train_dir;
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
                if (diannTask != null) {
                    diannTask.skip_check_file = diann_report_file;
                }

                if (tabbedPane != null) {
                    SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(tabbedPane.indexOfTab("Console")));
                }

                CmdTask[] initialTasks;
                if (conversionTask != null) {
                    initialTasks = new CmdTask[] { conversionTask, diannTask };
                } else {
                    initialTasks = new CmdTask[] { diannTask };
                }

                final String finalEffectiveMsFile = effectiveMsFile;
                final boolean finalIsTimsTOF = isTimsTOF;
                executeChainedCommands(initialTasks, () -> {
                    final CmdTask[] commandContainer = new CmdTask[1];
                    try {
                        SwingUtilities.invokeAndWait(() -> {
                            diannReportFileField.setText(diann_report_file);
                            // System.out.println("DIANN report file: " + diann_report_file);
                            CmdTask carafe_task = buildCarafeCommand(finalEffectiveMsFile, finalIsTimsTOF);
                            if (carafe_task != null) {
                                carafe_task.task_description = "Run Carafe to generate fine-tuned library";
                                if (!carafe_task.out_files.isEmpty()) {
                                    carafe_task.skip_check_file = carafe_task.out_files.getFirst();
                                }
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

                // Detect TIMSTOF data
                boolean isTimsTOF = false;
                // Check multi-file list directly (user may have selected .d folders)
                if (!trainMsFiles.isEmpty()) {
                    for (String msFile : trainMsFiles) {
                        if (msFile.toLowerCase().endsWith(".d")) {
                            File dFolder = new File(msFile);
                            File analysisTdf = new File(dFolder, "analysis.tdf");
                            if (dFolder.isDirectory() && analysisTdf.exists()) {
                                isTimsTOF = true;
                                break;
                            }
                        }
                    }
                }
                // Also check trainMsFile (single file or parent folder)
                if (!isTimsTOF) {
                    if (trainMsFile.toLowerCase().endsWith(".d") && new File(trainMsFile).isDirectory()) {
                        File analysisTdf = new File(trainMsFile, "analysis.tdf");
                        if (analysisTdf.exists()) {
                            isTimsTOF = true;
                        }
                    } else if (new File(trainMsFile).isDirectory()) {
                        // Check if parent folder contains .d subdirectories
                        File trainFileObj = new File(trainMsFile);
                        File[] dFolders = trainFileObj.listFiles((dir, name) -> name.toLowerCase().endsWith(".d"));
                        if (dFolders != null && dFolders.length > 0) {
                            for (File dFolder : dFolders) {
                                File analysisTdf = new File(dFolder, "analysis.tdf");
                                if (analysisTdf.exists()) {
                                    isTimsTOF = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                CmdTask carafeTask = buildCarafeCommand(null, isTimsTOF);
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
                        // Expand folder to individual files (RAW, mzML, or TimsTOF .d folders)
                        // check if there are mzML files in the folder first
                        File[] mzMLFiles = projectFile.listFiles((dir, name) -> name.toLowerCase().endsWith(".mzml"));
                        if (mzMLFiles != null && mzMLFiles.length > 0) {
                            for (File f : mzMLFiles) {
                                effectiveProjectFiles.add(f.getAbsolutePath());
                            }
                        } else {
                            // check if there are raw files in the folder
                            File[] rawFiles = projectFile.listFiles((dir, name) -> name.toLowerCase().endsWith(".raw"));
                            if (rawFiles != null && rawFiles.length > 0) {
                                for (File f : rawFiles) {
                                    effectiveProjectFiles.add(f.getAbsolutePath());
                                }
                            } else {
                                // check if there are TimsTOF .d folders in the folder
                                File[] dFolders = projectFile.listFiles(f -> 
                                    f.isDirectory() && f.getName().toLowerCase().endsWith(".d") 
                                    && new File(f, "analysis.tdf").exists());
                                if (dFolders != null) {
                                    for (File f : dFolders) {
                                        effectiveProjectFiles.add(f.getAbsolutePath());
                                    }
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
                boolean isTimsTofFolder = false;
                boolean isTimsTOFData = false;
                if (effectiveTrainFiles.size() == 1) {
                    File trainFile = new File(effectiveTrainFiles.get(0));
                    if (trainFile.isDirectory()) {
                        // Check if it's a TIMSTOF .d folder
                        if (trainFile.getName().toLowerCase().endsWith(".d")) {
                            File analysisTdf = new File(trainFile, "analysis.tdf");
                            if (analysisTdf.exists()) {
                                isTimsTofFolder = true;
                                isTimsTOFData = true;
                            }
                        }
                        if (!isTimsTofFolder) {
                            isDirectory = true;
                        }
                    }
                }

                // Check if we have TIMSTOF folders in the file list (multi-select case)
                boolean hasTimsTof = effectiveTrainFiles.stream().anyMatch(f -> {
                    File file = new File(f);
                    return f.toLowerCase().endsWith(".d") && file.isDirectory()
                            && new File(file, "analysis.tdf").exists();
                });

                if (isTimsTofFolder || hasTimsTof) {
                    // TIMSTOF .d folders - pass directly to DIA-NN, no conversion needed
                    isTimsTOFData = true;
                    finalTrainMzMLFiles.addAll(effectiveTrainFiles);
                } else if (isDirectory) {
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
                            conversionTask.out_dir = subDir;

                            for (File raw : rawFiles) {
                                conversionTask.input_files.add(raw.getAbsolutePath());
                                String rawName = raw.getName();
                                String baseName = rawName.lastIndexOf('.') > 0
                                        ? rawName.substring(0, rawName.lastIndexOf('.'))
                                        : rawName;
                                finalTrainMzMLFiles.add(subDir + File.separator + baseName + ".mzML");
                            }
                            singleEffectiveMsFile = subDir; // Pass subdir for searching
                        } else {
                            // Check for .d subdirectories (TIMSTOF batch folder)
                            File[] dFolders = trainFileObj.listFiles((d, n) -> n.toLowerCase().endsWith(".d"));
                            if (dFolders != null && dFolders.length > 0) {
                                for (File dFolder : dFolders) {
                                    File analysisTdf = new File(dFolder, "analysis.tdf");
                                    if (analysisTdf.exists()) {
                                        isTimsTOFData = true;
                                        finalTrainMzMLFiles.add(dFolder.getAbsolutePath());
                                    }
                                }
                            }
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
                        conversionTask.out_dir = subDir;
                        conversionTask.input_files.addAll(effectiveTrainFiles);

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

                // If RAW conversion is scheduled, the first converted mzML marks its completion.
                if (conversionTask != null && !finalTrainMzMLFiles.isEmpty()) {
                    conversionTask.skip_check_file = finalTrainMzMLFiles.getFirst();
                }

                CmdTask diannTask;
                if (!finalTrainMzMLFiles.isEmpty()) {
                    diannTask = buildDIANNCommand(finalTrainMzMLFiles, "", trainDb, diann_train_dir);
                } else {
                    diannTask = buildDIANNCommand(singleEffectiveMsFile, "", trainDb, diann_train_dir,
                            conversionTask != null);
                }

                String diann_report_file;
                if (this.isDiannV2) {
                    diann_report_file = diann_train_dir + File.separator + "report.parquet";
                } else {
                    diann_report_file = diann_train_dir + File.separator + "report.tsv";
                }
                if (diannTask != null) {
                    diannTask.task_description = "Run DIA-NN search on the training MS data";
                    diannTask.out_dir = diann_train_dir;
                    diannTask.out_files.add(diann_report_file);
                    diannTask.out_files_description.add("DIA-NN report");
                    diannTask.skip_check_file = diann_report_file;
                }

                String diann_project_dir = outDir + File.separator + "diann_project";
                File diannProjectDirFile = new File(diann_project_dir);
                if (!diannProjectDirFile.exists()) {
                    diannProjectDirFile.mkdirs();
                }
                final String carafeLibraryPath = outDir + File.separator + "carafe_library" + File.separator
                        + "carafe_spectral_library.tsv";

                if (tabbedPane != null) {
                    SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(tabbedPane.indexOfTab("Console")));
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
                final boolean finalIsTimsTOFData = isTimsTOFData;

                // Check if benchmark mode is enabled
                final boolean runBenchmark = benchmarkCheckbox != null && benchmarkCheckbox.isSelected();
                final String diann_libfree_dir = outDir + File.separator + "diann_project_library_free";

                executeChainedCommands(initialTasks, () -> {
                    final int numCommands = runBenchmark ? 3 : 2;
                    final CmdTask[] commands = new CmdTask[numCommands];
                    try {
                        SwingUtilities.invokeAndWait(() -> {
                            diannReportFileField.setText(diann_report_file);
                            CmdTask carafe_task = buildCarafeCommand(finalCarafeMsInput, finalIsTimsTOFData);
                            if (carafe_task != null) {
                                carafe_task.task_description = "Run Carafe to generate fine-tuned library";
                                if (!carafe_task.out_files.isEmpty()) {
                                    carafe_task.skip_check_file = carafe_task.out_files.getFirst();
                                }
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
                                commands[1].skip_check_file = diann_project_dir + File.separator + "report"
                                        + (isDiannV2 ? ".parquet" : ".tsv");
                            }

                            // Optional benchmark step: DIA-NN library-free search
                            if (runBenchmark && !effectiveProjectFiles.isEmpty()) {
                                // Create lib-free output directory
                                File libfreeDir = new File(diann_libfree_dir);
                                if (!libfreeDir.exists()) {
                                    libfreeDir.mkdirs();
                                }
                                // Build library-free command: pass empty library with database
                                commands[2] = buildDIANNCommand(effectiveProjectFiles, "", libraryDb,
                                        diann_libfree_dir);
                                if (commands[2] != null) {
                                    commands[2].task_description = "DIA-NN library-free search for benchmark comparison";
                                    commands[2].skip_check_file = diann_libfree_dir + File.separator + "report"
                                            + (isDiannV2 ? ".parquet" : ".tsv");
                                }
                            }
                        });
                    } catch (Exception e) {
                        e.printStackTrace();
                        return new CmdTask[0];
                    }
                    return commands;
                });
            }

            case 3, 4 -> { // Osprey workflows 4 & 5
                boolean endToEnd = (workflow == 4);
                String trainDb = trainDbFileField.getText().trim();
                String libraryDb = libraryDbFileField.getText().trim();
                boolean koina = koinaModelsFor((String) initialLibraryPredictorCombo.getSelectedItem()) != null;

                // Resolve training + project MS inputs; each non-mzML file becomes its own MSConvert
                // task so they can run as parallel processes.
                java.util.List<CmdTask> trainConv = new java.util.ArrayList<>();
                java.util.List<String> trainMs = resolveOspreyMsInputs(trainMsFiles, trainMsFileField.getText(),
                        outDir + File.separator + "train_mzML", trainConv);
                if (trainMs.isEmpty()) {
                    JOptionPane.showMessageDialog(this, "No valid training MS files (mzML/.d) found.",
                            "Input Required", JOptionPane.WARNING_MESSAGE);
                    setInputsFrozen(false);
                    return;
                }
                java.util.List<CmdTask> projConv = new java.util.ArrayList<>();
                java.util.List<String> projMs = null;
                if (endToEnd) {
                    projMs = resolveOspreyMsInputs(projectMsFiles, projectMsFileField.getText(),
                            outDir + File.separator + "project_mzML", projConv);
                    if (projMs.isEmpty()) {
                        JOptionPane.showMessageDialog(this, "No valid project MS files (mzML/.d) found.",
                                "Input Required", JOptionPane.WARNING_MESSAGE);
                        setInputsFrozen(false);
                        return;
                    }
                }

                boolean isTimsTOF = trainMs.stream().anyMatch(p -> p.toLowerCase().endsWith(".d"));
                String trainMsInput = trainMs.size() == 1 ? trainMs.getFirst()
                        : new File(trainMs.getFirst()).getParent();

                // When the training and library databases are the same file, build the peptide
                // FASTA + manifest once and reuse it for the new (finetuned) library.
                boolean sameDb;
                try {
                    sameDb = new File(trainDb).getCanonicalPath().equals(new File(libraryDb).getCanonicalPath());
                } catch (IOException ioe) {
                    sameDb = trainDb.equals(libraryDb);
                }

                // Deterministic output paths.
                String pep1 = outDir + File.separator + "osprey_train_db_peptides.fasta";
                String man1 = outDir + File.separator + "osprey_train_db_pairing.tsv";
                String lib1Tsv = outDir + File.separator + "osprey_initial_library"
                        + File.separator + "carafe_spectral_library.tsv";
                String ospreyTrainDir = outDir + File.separator + "osprey_train";
                String blib1 = ospreyTrainDir + File.separator + "osprey.blib";
                String lib2Tsv = outDir + File.separator + "osprey_new_library"
                        + File.separator + "carafe_spectral_library.tsv";
                // Entrapment peptides go ONLY into the library-DB FASTA, which feeds the finetuned
                // library (used as the project-search library in workflow 5, and the deliverable in
                // workflow 4). They must NOT go into the training-DB FASTA: the training search
                // drives AI fine-tuning, and identifying random entrapment sequences would pollute
                // that training. Consequently, when entrapment is requested the training and library
                // FASTAs differ even if the two databases are the same file, so we cannot share one.
                boolean entrap = includeEntrapmentCheckbox.isSelected();
                OspreyFastaPlanner.Plan fastaPlan = OspreyFastaPlanner.plan(sameDb, entrap);
                boolean shareFasta = fastaPlan.shareTrainingFasta;
                // The library-DB peptide FASTA + manifest: reuse the training-DB ones only when the
                // databases are identical AND no entrapment is requested.
                String pep2 = shareFasta ? pep1 : outDir + File.separator + "osprey_library_db_peptides.fasta";
                String man2 = shareFasta ? man1 : outDir + File.separator + "osprey_library_db_pairing.tsv";

                // Build all tasks up front (output paths are deterministic). The training-DB FASTA is
                // always target+decoy only (never entrapment) so fine-tuning is not trained on it.
                CmdTask ent1 = buildEntrapmentFastaCommand(trainDb, pep1, man1, fastaPlan.trainingEntrapment);
                ent1.task_description = "Build target-decoy peptide FASTA (training DB)";

                CmdTask lib1;
                if (koina) {
                    lib1 = buildKoinaLibraryCommand(pep1, lib1Tsv, trainMs.get(0));
                    if (lib1 != null) {
                        lib1.task_description = "Koina: predict initial target-decoy library";
                    }
                } else {
                    clearCarafeOverrides();
                    carafeDbOverride = pep1;
                    carafeEnzymeOverride = "NoCut";
                    carafeOutSubdirOverride = "osprey_initial_library";
                    carafeLfTypeOverride = "DIA-NN";
                    carafeSeOverride = "Osprey";
                    lib1 = buildCarafeCommand("", false); // empty -ms => library only
                    clearCarafeOverrides();
                    if (lib1 != null) {
                        lib1.task_description = "Carafe: build initial target-decoy library (NoCut)";
                    }
                }
                if (lib1 == null) {
                    setInputsFrozen(false);
                    return;
                }
                lib1.skip_check_file = lib1Tsv;

                new File(ospreyTrainDir).mkdirs();
                CmdTask osprey1 = buildOspreyCommand(trainMs, lib1Tsv, man1, ospreyTrainDir, null);
                if (osprey1 == null) {
                    setInputsFrozen(false);
                    return;
                }
                osprey1.task_description = "Osprey: search training files";

                // The library-DB FASTA carries the entrapment peptides (when requested). Build it
                // separately whenever it can't be shared with the training-DB FASTA (different DBs,
                // or entrapment requested). When shared, the training-DB target+decoy FASTA is reused.
                CmdTask ent2 = shareFasta ? null
                        : buildEntrapmentFastaCommand(libraryDb, pep2, man2, fastaPlan.libraryEntrapment);
                if (ent2 != null) {
                    ent2.task_description = entrap
                            ? "Build target-decoy-entrapment peptide FASTA (library DB)"
                            : "Build target-decoy peptide FASTA (library DB)";
                    if (sameDb && entrap) {
                        logToConsole("[Carafe] Entrapment enabled: building a separate library "
                                + "peptide FASTA WITH entrapment for the finetuned library, while the "
                                + "training FASTA stays entrapment-free so fine-tuning is not trained "
                                + "on entrapment sequences.\n");
                    }
                } else {
                    logToConsole("[Carafe] Training and library databases are identical (no entrapment); "
                            + "reusing the same peptide FASTA + manifest for both libraries.\n");
                }

                clearCarafeOverrides();
                carafeDbOverride = pep2;
                carafeEnzymeOverride = "NoCut";
                carafeIOverride = blib1;
                carafeSeOverride = "Osprey";
                carafeOutSubdirOverride = "osprey_new_library";
                carafeLfTypeOverride = "DIA-NN";
                CmdTask lib2 = buildCarafeCommand(trainMsInput, isTimsTOF);
                clearCarafeOverrides();
                if (lib2 == null) {
                    setInputsFrozen(false);
                    return;
                }
                lib2.task_description = "Carafe: finetune on Osprey results and build new library";
                lib2.skip_check_file = lib2Tsv;

                CmdTask osprey2 = null;
                if (endToEnd) {
                    String ospreyProjectDir = outDir + File.separator + "osprey_project";
                    new File(ospreyProjectDir).mkdirs();
                    // When entrapment is on, have Osprey also write an FDRBench input TSV under
                    // osprey_project/FDRBench (level follows the Osprey tab's --fdr-level); the
                    // pairing manifest is copied alongside it after the search (see buildOspreyCommand).
                    String fdrbenchOut = OspreyFdrBenchPlanner.fdrBenchInputPath(entrap, true, ospreyProjectDir);
                    osprey2 = buildOspreyCommand(projMs, lib2Tsv, man2, ospreyProjectDir, fdrbenchOut);
                    if (osprey2 == null) {
                        setInputsFrozen(false);
                        return;
                    }
                    osprey2.task_description = "Osprey: search project files with the new library";
                }

                // ---- Phases ----
                // Phase 1 (parallel): MSConvert lanes (throttled), the library-DB entrapment FASTA,
                // and the training-DB entrapment FASTA. With Koina, the initial-library prediction
                // (remote) runs in the same lane right after its FASTA, concurrently with MSConvert.
                // With local Carafe, the library prediction is deferred to Phase 2 so it does not
                // compete with MSConvert for local CPU/GPU.
                // When "auto" NCE is selected and inputs must be converted, the Koina library needs
                // the mzML to read the collision energy, so it cannot run before conversion; defer
                // it to Phase 2. Otherwise (manual NCE, or mzML inputs) the Koina lane runs in
                // Phase 1 concurrently with conversion.
                boolean autoNce = nceField.getText().trim().equalsIgnoreCase("auto");
                boolean hasConversion = !trainConv.isEmpty() || !projConv.isEmpty();
                boolean koinaParallel = koina && !(autoNce && hasConversion);

                java.util.List<java.util.List<CmdTask>> phase1 = new java.util.ArrayList<>();
                for (CmdTask c : trainConv) {
                    phase1.add(java.util.List.of(c));
                }
                for (CmdTask c : projConv) {
                    phase1.add(java.util.List.of(c));
                }
                if (ent2 != null) {
                    phase1.add(java.util.List.of(ent2));
                }
                if (koinaParallel) {
                    phase1.add(java.util.Arrays.asList(ent1, lib1));
                } else {
                    phase1.add(java.util.List.of(ent1));
                }

                // Phase 2 (sequential): the deferred library (local, or Koina waiting for mzML for
                // auto NCE), then Osprey search, finetune, re-search.
                java.util.List<CmdTask> seq = new java.util.ArrayList<>();
                if (!koinaParallel) {
                    seq.add(lib1);
                }
                seq.add(osprey1);
                seq.add(lib2);
                if (endToEnd && osprey2 != null) {
                    seq.add(osprey2);
                }

                java.util.List<java.util.List<java.util.List<CmdTask>>> phases = new java.util.ArrayList<>();
                phases.add(phase1);
                phases.add(java.util.List.of(seq));

                int convThreads = 4;
                try {
                    convThreads = Integer.parseInt(ospreyConversionThreadsField.getText().trim());
                } catch (NumberFormatException ignore) {
                    // keep default
                }
                if (convThreads < 1) {
                    convThreads = 1;
                }

                // MSConvert settings are identical for every file, so log them once here instead of
                // repeating the verbose per-file header in the console.
                int nConvert = trainConv.size() + projConv.size();
                if (nConvert > 0) {
                    logToConsole("\n[MSConvert] Converting " + nConvert + " file(s) to indexed mzML "
                            + "(peak picking MS1-2, zlib compression) using up to " + convThreads
                            + " parallel process(es). Per-file headers are suppressed; only the output "
                            + "file and any errors are shown.\n");
                }

                if (tabbedPane != null) {
                    SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(tabbedPane.indexOfTab("Console")));
                }
                executeParallelWorkflow(phases, convThreads);
            }

            default -> JOptionPane.showMessageDialog(this, "Unsupported workflow selected!", "Error",
                    JOptionPane.ERROR_MESSAGE);
        }
    }

    @FunctionalInterface
    private interface NextCommandsSupplier {
        CmdTask[] getNextCommands();
    }

    /**
     * If "Reuse existing results" is enabled and the command's expected output already
     * exists, record the step as skipped and return true. Otherwise return false, meaning
     * the caller should run the command normally.
     */
    private boolean skipIfResultPresent(CmdTask command, boolean reuseResults, int stepIndex) {
        if (!reuseResults || command == null || command.skip_check_file == null
                || !new File(command.skip_check_file).exists()) {
            return false;
        }
        // The output exists, but only reuse it if the step's command + inputs are unchanged.
        // Computed here (not at build time) so chained inputs from upstream steps already exist,
        // giving the correct cascade: a changed upstream output re-runs every downstream step.
        String signature = main.java.util.ReuseSignature.compute(
                command.args, command.cmd, command.input_files);
        if (!main.java.util.ReuseSignature.matches(command.skip_check_file, signature)) {
            logToConsole("\n========================================\n");
            logToConsole("[RE-RUN] " + command.task_description + "\n");
            logToConsole("Inputs or settings changed since the last run — re-running.\n");
            logToConsole("========================================\n\n");
            return false;
        }
        command.skipped = true;
        command.time_used = 0.0;
        String now = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
        command.time_start = now;
        command.time_end = now;
        logToConsole("\n========================================\n");
        logToConsole("[SKIP] " + command.task_description + "\n");
        logToConsole("Reusing existing result: " + command.skip_check_file + "\n");
        logToConsole("========================================\n\n");
        String key = String.format("%02d. %s - %s (skipped)", stepIndex, command.task_name,
                command.task_description);
        timeUsageMap.put(key, 0.0);
        tasks.add(command);
        return true;
    }

    /**
     * Persist the reuse signature next to a step's output after it completes successfully, so a
     * later run with the same command and unchanged inputs can be skipped. No-op when the step has
     * no {@code skip_check_file} (steps that are never auto-reused).
     */
    private void writeStepSignature(CmdTask command) {
        if (command == null || command.skip_check_file == null) {
            return;
        }
        if (!new File(command.skip_check_file).exists()) {
            return;
        }
        String signature = main.java.util.ReuseSignature.compute(
                command.args, command.cmd, command.input_files);
        String normalizedCmd = !command.args.isEmpty()
                ? StringUtils.join(command.args, " ")
                : command.cmd;
        main.java.util.ReuseSignature.writeSidecar(
                command.skip_check_file, signature, normalizedCmd, command.input_files);
    }

    private void executeChainedCommands(CmdTask[] initialCommands, NextCommandsSupplier nextCommandsSupplier) {
        isRunning = true;
        runButton.setEnabled(false);
        stopButton.setEnabled(true);
        progressBar.setIndeterminate(true);
        progressBar.setString("Running DIA-NN...");

        timeUsageMap.clear();
        tasks.clear();

        final boolean reuseResults = reuseResultsCheckbox != null && reuseResultsCheckbox.isSelected();

        Object selectedPython = pythonPathCombo.getSelectedItem();
        String pythonPath = selectedPython != null ? selectedPython.toString().trim() : "";
        if (!pythonPath.isEmpty()) {
            prefs.put(PREF_PYTHON_PATH, pythonPath);
        }

        // Automatically save screenshots of parameter panels at the start
        // Ensure this runs on EDT before background thread starts
        saveParameterScreenshots();
        autoSaveRunSettings();

        executor = Executors.newSingleThreadExecutor();
        executor.submit(() -> {
            try {
                int stepIndex = 1;
                for (CmdTask command : initialCommands) {
                    if (!isRunning)
                        return;

                    if (skipIfResultPresent(command, reuseResults, stepIndex)) {
                        stepIndex++;
                        continue;
                    }

                    updateProgressBarForCommand(command.task_description);

                    logToConsole("\n========================================\n");
                    logToConsole("Running: " + command.task_description + "\n");
                    logToConsole("Command: " + command.cmd + "\n");
                    logToConsole("========================================\n\n");
                    command.time_start = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
                    long start = System.nanoTime();
                    int exitCode = runSingleCommand(command, pythonPath);
                    long end = System.nanoTime();
                    double minutes = (end - start) / 1e9 / 60.0;
                    command.time_end = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
                    command.time_used = minutes;
                    String key = String.format("%02d. %s - %s", stepIndex++, command.task_name,
                            command.task_description);
                    timeUsageMap.put(key, minutes);
                    tasks.add(command);

                    if (exitCode != 0) {
                        SwingUtilities.invokeLater(() -> {
                            logToConsole("\n[ERROR] Command failed with exit code: " + exitCode + "\n");
                            progressBar.setString("Failed");
                            finishExecution();
                        });
                        return;
                    }
                    writeStepSignature(command);
                }

                if (nextCommandsSupplier != null && isRunning) {
                    CmdTask[] nextCommands = nextCommandsSupplier.getNextCommands();
                    for (CmdTask command : nextCommands) {
                        if (!isRunning)
                            return;

                        if (skipIfResultPresent(command, reuseResults, stepIndex)) {
                            stepIndex++;
                            continue;
                        }

                        updateProgressBarForCommand(command.task_description);

                        logToConsole("\n========================================\n");
                        logToConsole("Running: " + command.task_description + "\n");
                        logToConsole("Command: " + command.cmd + "\n");
                        logToConsole("========================================\n\n");
                        long start = System.nanoTime();
                        command.time_start = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
                        int exitCode = runSingleCommand(command, pythonPath);
                        long end = System.nanoTime();
                        double minutes = (end - start) / 1e9 / 60.0;
                        command.time_end = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
                        command.time_used = minutes;
                        String key = String.format("%02d. %s - %s", stepIndex++, command.task_name,
                                command.task_description);
                        timeUsageMap.put(key, minutes);
                        tasks.add(command);
                        if (exitCode != 0) {
                            SwingUtilities.invokeLater(() -> {
                                logToConsole("\n[ERROR] Command failed with exit code: " + exitCode + "\n");
                                progressBar.setString("Failed");
                                finishExecution();
                            });
                            return;
                        }
                        writeStepSignature(command);
                    }
                }

                SwingUtilities.invokeLater(() -> {
                    logToConsole("\n[SUCCESS] Workflow completed successfully!\n");
                    progressBar.setString("Completed");
                    logToConsole("\n[SUMMARY] Step durations (min):\n");
                    double totalTime = 0.0;
                    for (java.util.Map.Entry<String, Double> e : timeUsageMap.entrySet()) {
                        // logToConsole(" - " + e.getKey() + " : " + String.format("%.2f", e.getValue()) + "\n");
                        totalTime += e.getValue();
                    }
                    int taskIndex = 1;
                    for (CmdTask task : tasks) {
                        String skippedNote = task.skipped ? " (skipped — reused existing result)" : "";
                        logToConsole("\n[" + String.format("%02d", taskIndex++) + "] " + task.task_name + " - " + task.task_description + skippedNote + "\n");
                        // logToConsole("  Command: " + task.cmd + "\n");
                        logToConsole("  Output directory: " + task.out_dir + "\n");
                        if(!task.out_files.isEmpty()){
                            for(int k=0;k<task.out_files.size();k++){
                                logToConsole("  - " + task.out_files_description.get(k) + ": " + task.out_files.get(k) + "\n");
                            }
                        }
                        logToConsole("  Time start: " + task.time_start + "\n");
                        logToConsole("  Time end: " + task.time_end + "\n");
                        logToConsole("  Time used: " + String.format("%.2f", task.time_used) + " min\n");
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

    /** Prepend the Python executable's directory to PATH for a process (same as runSingleCommand). */
    private void applyPythonPathEnv(ProcessBuilder pb, String pythonPath) {
        if (pythonPath == null || pythonPath.isEmpty()) {
            return;
        }
        java.util.Map<String, String> env = pb.environment();
        File pythonFile = new File(pythonPath);
        String pythonDir = pythonFile.isFile() ? pythonFile.getParent() : pythonPath;
        if (pythonDir != null) {
            boolean isWindows = System.getProperty("os.name").toLowerCase().contains("windows");
            String sep = isWindows ? ";" : ":";
            String currentPath = env.getOrDefault("PATH", env.getOrDefault("Path", ""));
            String newPath = pythonDir + sep + currentPath;
            env.put("PATH", newPath);
            if (isWindows) {
                env.put("Path", newPath);
            }
        }
    }

    /**
     * Parallel-safe command runner: like {@link #runSingleCommand} but tracks its process in
     * {@link #activeProcesses} (so Stop can terminate it) instead of the shared
     * {@code currentProcess} field, and prefixes each output line with {@code prefix} so the
     * interleaved output of concurrent lanes stays readable.
     */
    private int runTrackedCommand(CmdTask task, String pythonPath, String prefix) throws Exception {
        ProcessBuilder pb = new ProcessBuilder();
        if (task.args != null && !task.args.isEmpty()) {
            pb.command(task.args);
        } else if (System.getProperty("os.name").toLowerCase().contains("windows")) {
            pb.command("cmd", "/c", "\"" + task.cmd + "\"");
        } else {
            pb.command("bash", "-c", task.cmd);
        }
        pb.redirectErrorStream(true);
        applyPythonPathEnv(pb, pythonPath);

        // MSConvert prints a verbose, identical header for every file; with many parallel
        // conversions that floods the console. Suppress the boilerplate and keep only meaningful
        // lines (the output-file line and any errors/warnings). The common settings are logged
        // once before the conversion phase.
        boolean isMsConvert = "MSConvert".equalsIgnoreCase(task.task_name);

        Process proc = pb.start();
        activeProcesses.add(proc);
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                if (isMsConvert && !showMsConvertLine(line)) {
                    continue;
                }
                logToConsole("[" + prefix + "] " + line + "\n");
            }
            return proc.waitFor();
        } finally {
            activeProcesses.remove(proc);
        }
    }

    /**
     * Whether an MSConvert output line is worth showing on the console. We keep the per-file
     * "writing output file" line and any error/warning/failure lines, and drop the repetitive
     * format/compression/filter header (logged once before the conversion phase).
     */
    private boolean showMsConvertLine(String line) {
        String l = line.toLowerCase();
        return l.contains("writing output file")
                || l.contains("error") || l.contains("warning")
                || l.contains("fail") || l.contains("exception");
    }

    /**
     * Execute a workflow as a sequence of phases. Within a phase, lanes run concurrently (barrier
     * at the end of the phase); each lane runs its tasks sequentially. MSConvert tasks are
     * throttled to {@code convConcurrency} concurrent processes via a semaphore, while other lanes
     * (e.g. a Koina library prediction running on a remote server) run freely alongside them.
     *
     * @param phases ordered phases; each phase is a list of lanes; each lane is a list of tasks
     * @param convConcurrency max number of concurrent MSConvert processes
     */
    private void executeParallelWorkflow(java.util.List<java.util.List<java.util.List<CmdTask>>> phases,
            int convConcurrency) {
        isRunning = true;
        runButton.setEnabled(false);
        stopButton.setEnabled(true);
        progressBar.setIndeterminate(true);
        progressBar.setString("Running...");
        timeUsageMap.clear();
        tasks.clear();
        activeProcesses.clear();

        final boolean reuseResults = reuseResultsCheckbox != null && reuseResultsCheckbox.isSelected();
        Object selectedPython = pythonPathCombo.getSelectedItem();
        final String pythonPath = selectedPython != null ? selectedPython.toString().trim() : "";
        if (!pythonPath.isEmpty()) {
            prefs.put(PREF_PYTHON_PATH, pythonPath);
        }
        saveParameterScreenshots();
        autoSaveRunSettings();

        final java.util.concurrent.Semaphore convSem =
                new java.util.concurrent.Semaphore(Math.max(1, convConcurrency));
        final java.util.concurrent.atomic.AtomicInteger stepCounter =
                new java.util.concurrent.atomic.AtomicInteger(1);
        final java.util.concurrent.atomic.AtomicBoolean failed =
                new java.util.concurrent.atomic.AtomicBoolean(false);

        executor = java.util.concurrent.Executors.newSingleThreadExecutor();
        executor.submit(() -> {
            try {
                for (java.util.List<java.util.List<CmdTask>> phase : phases) {
                    if (!isRunning || failed.get()) {
                        break;
                    }
                    java.util.concurrent.ExecutorService pool =
                            java.util.concurrent.Executors.newCachedThreadPool();
                    java.util.List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
                    for (java.util.List<CmdTask> lane : phase) {
                        futures.add(pool.submit(
                                () -> runLane(lane, pythonPath, reuseResults, stepCounter, convSem, failed)));
                    }
                    pool.shutdown();
                    for (java.util.concurrent.Future<?> fut : futures) {
                        try {
                            fut.get();
                        } catch (Exception e) {
                            failed.set(true);
                        }
                    }
                    if (failed.get()) {
                        break;
                    }
                }

                SwingUtilities.invokeLater(() -> {
                    if (!isRunning || failed.get()) {
                        progressBar.setString("Failed");
                    } else {
                        logToConsole("\n[SUCCESS] Workflow completed successfully!\n");
                        progressBar.setString("Completed");
                        printWorkflowSummary();
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

    /** Run one lane's tasks sequentially; sets {@code failed} on the first non-zero exit/error. */
    private void runLane(java.util.List<CmdTask> lane, String pythonPath, boolean reuseResults,
            java.util.concurrent.atomic.AtomicInteger stepCounter, java.util.concurrent.Semaphore convSem,
            java.util.concurrent.atomic.AtomicBoolean failed) {
        for (CmdTask command : lane) {
            if (!isRunning || failed.get()) {
                return;
            }
            int stepIndex = stepCounter.getAndIncrement();
            if (skipIfResultPresent(command, reuseResults, stepIndex)) {
                continue;
            }
            String prefix = command.task_name;
            boolean isConversion = "MSConvert".equalsIgnoreCase(command.task_name);
            int exitCode;
            long start = System.nanoTime();
            try {
                // For conversions, wait for a free slot BEFORE announcing the start, so the
                // [START] markers reflect files that are actually converting (not all queued at once).
                if (isConversion) {
                    convSem.acquire();
                }
                if (!isRunning || failed.get()) {
                    if (isConversion) {
                        convSem.release();
                    }
                    return;
                }
                logToConsole("\n=== [START] " + command.task_description + " ===\n");
                command.time_start = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
                start = System.nanoTime();
                try {
                    exitCode = runTrackedCommand(command, pythonPath, prefix);
                } finally {
                    if (isConversion) {
                        convSem.release();
                    }
                }
            } catch (Exception e) {
                logToConsole("\n[ERROR] " + command.task_description + " failed: " + e.getMessage() + "\n");
                failed.set(true);
                return;
            }
            long end = System.nanoTime();
            command.time_used = (end - start) / 1e9 / 60.0;
            command.time_end = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
            timeUsageMap.put(String.format("%02d. %s - %s", stepIndex, command.task_name,
                    command.task_description), command.time_used);
            tasks.add(command);
            logToConsole("\n=== [DONE] " + command.task_description + " ("
                    + String.format("%.2f", command.time_used) + " min) ===\n");
            if (exitCode != 0) {
                logToConsole("\n[ERROR] " + command.task_description + " failed with exit code: "
                        + exitCode + "\n");
                failed.set(true);
                return;
            }
            // Post-success hook (e.g. move a locally-staged blib to its final network path).
            // Runs only on a clean exit and before the reuse signature is written.
            if (command.postAction != null) {
                try {
                    command.postAction.run();
                } catch (Exception ex) {
                    logToConsole("\n[ERROR] " + command.task_description + " post-step failed: "
                            + ex.getMessage() + "\n");
                    failed.set(true);
                    return;
                }
            }
            writeStepSignature(command);
        }
    }

    /** Print the per-step duration summary (shared by the parallel executor). */
    private void printWorkflowSummary() {
        logToConsole("\n[SUMMARY] Step durations (min):\n");
        double totalTime = 0.0;
        synchronized (timeUsageMap) {
            for (java.util.Map.Entry<String, Double> e : timeUsageMap.entrySet()) {
                totalTime += e.getValue();
            }
        }
        int taskIndex = 1;
        synchronized (tasks) {
            for (CmdTask task : tasks) {
                String skippedNote = task.skipped ? " (skipped — reused existing result)" : "";
                logToConsole("\n[" + String.format("%02d", taskIndex++) + "] " + task.task_name + " - "
                        + task.task_description + skippedNote + "\n");
                logToConsole("  Output directory: " + task.out_dir + "\n");
                if (!task.out_files.isEmpty()) {
                    for (int k = 0; k < task.out_files.size(); k++) {
                        logToConsole("  - " + task.out_files_description.get(k) + ": " + task.out_files.get(k) + "\n");
                    }
                }
                logToConsole("  Time used: " + String.format("%.2f", task.time_used) + " min\n");
            }
        }
        logToConsole("Total time: " + String.format("%.2f", totalTime) + " min\n");
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
            updateDIANNpermission(diann_path);
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

            boolean isTimsTOFData = false;
            for (String f : msFiles) {
                diannArgs.add("--f");
                diannArgs.add(f);
                // diannArgs.add("\"" + f + "\"");

                // check if this is a TimsTOF .d folder (i.e., timsTOF DIA data)
                if (GenericUtils.isTimsTOFData(f)) {
                    isTimsTOFData = true;
                }
            }

            if (spectral_library_file.isEmpty() && !database.isEmpty()) {
                if (database.endsWith(".speclib")) {
                    diannArgs.add("--lib");
                    diannArgs.add(database);
                    // diannArgs.add("\"" + spectral_library_file + "\"");  
                    diannArgs.add("--gen-spec-lib");
                } else {
                    // .fasta or .fa
                    diannArgs.add("--lib");
                    diannArgs.add("");
                    // diannArgs.add("\"\"");
                    diannArgs.add("--gen-spec-lib");
                    diannArgs.add("--predictor");
                    diannArgs.add("--fasta");
                    diannArgs.add(database);
                    // diannArgs.add("\"" + database + "\"");
                    diannArgs.add("--fasta-search");
                }
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
            // remove "," at the start and end
            fixModSelected = fixModSelected.replaceAll("^,", "");
            fixModSelected = fixModSelected.replaceAll(",$", "");
            if (fixModSelected.equalsIgnoreCase("1")) {
                diannArgs.add("--unimod4");
            }else if (fixModSelected.equalsIgnoreCase("0")) {
                // console output
                consoleArea.append("Fixed modifications disabled\n");
            } else {
                JOptionPane.showMessageDialog(this,
                        "Unsupported modification settings. Please select '1' for Fixed modifications.", "Warning",
                        JOptionPane.WARNING_MESSAGE);
                return null;
            }

            String varModSelected = varModSelectedField.getText().trim();
            // remove "," at the start and end
            varModSelected = varModSelected.replaceAll("^,", "");
            varModSelected = varModSelected.replaceAll(",$", "");
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
            } else if (varModSelected.equalsIgnoreCase("7,8,9")) {
                // 1.8.1: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "C:\tools\DIA-NN\1.8.1\report.tsv" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --monitor-mod UniMod:21
                // --reanalyse --relaxed-prot-inf --smart-profiling --peak-center
                // --no-ifs-removal
                // 1.9.1: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "D:\software\DIA-NN\1.9.1/report.tsv" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms
                // --relaxed-prot-inf --rt-profiling
                // 1.9.2: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "D:\software\DIA-NN\1.9.2/report.tsv" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms
                // --relaxed-prot-inf --rt-profiling
                // 2.0: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "D:\software\DIA-NN\2.0\report.parquet" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse
                // --rt-profiling
                // 2.0.2: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "D:\software\DIA-NN\2.0.2\report.parquet" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse
                // --rt-profiling
                // 2.1.0: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "D:\software\DIA-NN\2.1.0\report.parquet" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse
                // --rt-profiling
                // 2.2.0: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "D:\software\DIA-NN\2.2.0\report.parquet" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse
                // --rt-profiling
                // 2.2.1: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "D:\software\DIA-NN\2.2.1\report.parquet" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse
                // --rt-profiling
                // 2.3.0: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "D:\software\DIA-NN\2.3.0\report.parquet" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse
                // --rt-profiling
                // 2.3.1: diann.exe --lib "" --threads 8 --verbose 1 --out
                // "D:\software\DIA-NN\2.3.1\report.parquet" --qvalue 0.01 --matrices --unimod4
                // --var-mods 1 --var-mod UniMod:21,79.966331,STY --peptidoforms --reanalyse
                // --rt-profiling
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
            } else if (varModSelected.equalsIgnoreCase("5")) {
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
                // --var-mod UniMod:1,42.010565,*n
                diannArgs.add("UniMod:1,42.010565,*n");
                diannArgs.add("--peptidoforms");
            } else if (varModSelected.equalsIgnoreCase("2,5") || varModSelected.equalsIgnoreCase("5,2")) {
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
                diannArgs.add("--var-mod");
                // --var-mod UniMod:1,42.010565,*n
                diannArgs.add("UniMod:1,42.010565,*n");
                diannArgs.add("--peptidoforms");
            } else if (varModSelected.equalsIgnoreCase("3,4") || varModSelected.equalsIgnoreCase("4,3")) {
                // This modification is not supported in 1.8.1, 1.9.1, 1.9.2, 2.0, 2.0.2, 2.1.0, 2.2.0
                // It is supported started from 2.3.0
                // --var-mods 1 --var-mod UniMod:7,0.984016,NQ
                if(diannVersion.startsWith("2.3")) {
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
                    diannArgs.add("UniMod:7,0.984016,NQ");
                    diannArgs.add("--peptidoforms");
                }else{
                    // show warning message
                    JOptionPane.showMessageDialog(this,
                            "The selected variable modification (deamidation on N/Q) is only supported in DIANN version 2.3.0 or later. Please select DIANN version 2.3.0 or later to use this modification. The current DIANN version is: " + diannVersion,
                            "Warning", JOptionPane.WARNING_MESSAGE);
                    return null;
                }
            } else {
                if (isV2) {
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
                    String[] var_int_list = varModSelected.split(",");
                    ArrayList<String> phos_var_mods = new ArrayList<>();
                    // For N/Q
                    ArrayList<String> deamidation_var_mods = new ArrayList<>();
                    for (String var_int : var_int_list) {
                        if (var_int.trim().equalsIgnoreCase("0")) {
                            // no modification
                        } else if (var_int.trim().equalsIgnoreCase("2")) {
                            diannArgs.add("--var-mod");
                            diannArgs.add("UniMod:35,15.994915,M");
                        } else if (var_int.trim().equalsIgnoreCase("3")) {
                            deamidation_var_mods.add("N");
                        } else if (var_int.trim().equalsIgnoreCase("4")) {
                            deamidation_var_mods.add("Q");
                        } else if (var_int.trim().equalsIgnoreCase("5")) {
                            diannArgs.add("--var-mod");
                            diannArgs.add("UniMod:1,42.010565,*n");
                        } else if (var_int.trim().equalsIgnoreCase("7")) {
                            phos_var_mods.add("S");
                        } else if (var_int.trim().equalsIgnoreCase("8")) {
                            phos_var_mods.add("T");
                        } else if (var_int.trim().equalsIgnoreCase("9")) {
                            phos_var_mods.add("Y");
                        } else if (var_int.trim().equalsIgnoreCase("10")) {
                            diannArgs.add("--var-mod");
                            diannArgs.add("UniMod:121,114.042927,K");
                            diannArgs.add("--no-cut-after-mod");
                            diannArgs.add("UniMod:121");
                        } else {
                            JOptionPane.showMessageDialog(this,
                                    "Unsupported modification settings. Please select '2' for Variable modifications.",
                                    "Warning",
                                    JOptionPane.WARNING_MESSAGE);
                            return null;
                        }
                    }
                    if (!phos_var_mods.isEmpty()) {
                        diannArgs.add("--var-mod");
                        diannArgs.add("UniMod:21,79.966331," + String.join(",", phos_var_mods));
                    }
                    if (!deamidation_var_mods.isEmpty()) {
                        diannArgs.add("--var-mod");
                        // --var-mods 1 --var-mod UniMod:7,0.984016,NQ
                        diannArgs.add("UniMod:7,0.984016," + String.join(",", deamidation_var_mods));
                    }
                    diannArgs.add("--peptidoforms");
                }
            }
            
            String [] d_enzyme = ((String) enzymeCombo.getSelectedItem()).split(": ");
            String enzyme = d_enzyme[0];
            String enzymeName = d_enzyme[1];
            if (enzymeName.equals("Trypsin")) {
                // DIANN Trypsin: --cut K*,R*,!*P
                diannArgs.add("--cut");
                diannArgs.add("K*,R*,!*P");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else if (enzymeName.equals("Trypsin (no P rule)")) {
                // DIANN Trypsin/P (default): --cut K*,R*
                diannArgs.add("--cut");
                diannArgs.add("K*,R*");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else if (enzymeName.equals("Lys-C (no P rule)")) {
                // DIANN Lys-C: --cut K*
                diannArgs.add("--cut");
                diannArgs.add("K*");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else if (enzymeName.equals("Chymotrypsin")) {
                // DIANN Chymotrypsin: --cut F*,Y*,W*,M*,L*,!*P
                diannArgs.add("--cut");
                diannArgs.add("F*,Y*,W*,M*,L*,!*P");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else if (enzymeName.equals("Asp-N")) {
                // DIANN Asp-N: --cut *D
                diannArgs.add("--cut");
                diannArgs.add("*D");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else if (enzymeName.equals("Glu-C")) {
                // DIANN Glu-C: --cut E*
                diannArgs.add("--cut");
                diannArgs.add("E*");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else if (enzymeName.equals("Arg-C (no P rule)")) {
                // DIANN Arg-C: --cut R*
                diannArgs.add("--cut");
                diannArgs.add("R*");
                diannArgs.add("--missed-cleavages");
                diannArgs.add(String.valueOf(missCleavageSpinner.getValue()));
            } else if (enzymeName.equals("non-specific")){
                // DIANN non-specific: --cut ** --missed-cleavages 100
                diannArgs.add("--cut");
                diannArgs.add("**");
                diannArgs.add("--missed-cleavages");
                diannArgs.add("100");
            } else {
                JOptionPane.showMessageDialog(this,
                        "Unsupported enzyme setting for DIANN. Please select a supported enzyme.",
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

            if (isTimsTOFData) {
                // https://github.com/vdemichev/DiaNN?tab=readme-ov-file#changing-default-settings
                // add --mass-acc 15 --mass-acc-ms1 15
                diannArgs.add("--mass-acc");
                diannArgs.add("15");
                diannArgs.add("--mass-acc-ms1");
                diannArgs.add("15");
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
                            // also show this to console panel
                            logToConsole("The additional DIA-NN option " + option + " is redundant!\n");
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
            task.input_files.addAll(msFiles);
            if (database != null && !database.isEmpty()) {
                task.input_files.add(database);
            }
            if (spectral_library_file != null && !spectral_library_file.isEmpty()) {
                task.input_files.add(spectral_library_file);
            }
            task.out_dir = out_dir;
            return task;
        } else {
            JOptionPane.showMessageDialog(this, "Please provide a valid DIA-NN executable path.", "Input Required",
                    JOptionPane.WARNING_MESSAGE);
            return null;
        }
    }

    /**
     * Update the executable permission of the DIA-NN executable.
     * @param diannPath The path to the DIA-NN executable.
     */
    private void updateDIANNpermission(String diannPath){
        String osName = System.getProperty("os.name").toLowerCase();
        boolean isWindows = osName.contains("win");
        File f = new File(diannPath);
        if (f.exists()) {
            // Set executable permission on Linux/macOS
            if (!isWindows && !f.canExecute()) {
                boolean success = f.setExecutable(true);
                if (success) {
                    Cloger.getInstance().logger.info("Set executable permission for: " + diannPath);
                } else {
                    Cloger.getInstance().logger.warn("Failed to set executable permission for: " + diannPath);
                }
            }
            Cloger.getInstance().logger.info("DIANN found at: " + diannPath);
        }else{
            Cloger.getInstance().logger.error("DIANN not found at: " + diannPath);
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

    /** The search engine selected in the GUI, defaulting to DIA-NN. */
    private String getSelectedSearchEngine() {
        if (searchEngineCombo != null && searchEngineCombo.getSelectedItem() != null) {
            return searchEngineCombo.getSelectedItem().toString();
        }
        return "DIA-NN";
    }

    /** The .NET runtime identifier for the current platform (used to find the bundled Osprey). */
    private String getOspreyRid() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        if (os.contains("win")) {
            return "win-x64";
        } else if (os.contains("mac") || os.contains("darwin")) {
            return (arch.contains("aarch64") || arch.contains("arm")) ? "osx-arm64" : "osx-x64";
        } else {
            return "linux-x64";
        }
    }

    /**
     * Resolve the Osprey executable. Checks, in order: the saved preference, the bundled
     * location next to the Carafe jar ({@code osprey/<rid>/}), {@code ~/.carafe/osprey/<rid>/},
     * then the system PATH. Returns the first existing path, or an empty string if none is found.
     */
    private String resolveOspreyBinary() {
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
        String exeName = isWindows ? "Osprey.exe" : "Osprey";
        String rid = getOspreyRid();

        // 1. Saved preference (and the editable combo if present).
        String saved = prefs.get(PREF_OSPREY_PATH, "");
        if (ospreyPathCombo != null && ospreyPathCombo.getSelectedItem() != null) {
            String fromCombo = ospreyPathCombo.getSelectedItem().toString().trim();
            if (!fromCombo.isEmpty()) {
                saved = fromCombo;
            }
        }
        if (!saved.isEmpty() && new File(saved).isFile()) {
            return saved;
        }

        // 2. Bundled next to the Carafe jar: <jarDir>/osprey/<rid>/Osprey(.exe).
        try {
            File jar = new File(getCarafeJarPath());
            File jarDir = jar.getParentFile();
            if (jarDir != null) {
                File bundled = new File(jarDir, "osprey" + File.separator + rid + File.separator + exeName);
                if (bundled.isFile()) {
                    return bundled.getAbsolutePath();
                }
            }
        } catch (Exception ignore) {
            // fall through to other locations
        }

        // 3. ~/.carafe/osprey/<rid>/Osprey(.exe).
        File home = new File(System.getProperty("user.home"),
                ".carafe" + File.separator + "osprey" + File.separator + rid + File.separator + exeName);
        if (home.isFile()) {
            return home.getAbsolutePath();
        }

        // 4. System PATH.
        try {
            String which = isWindows ? "where" : "which";
            Process p = new ProcessBuilder(which, exeName).redirectErrorStream(true).start();
            try (java.io.BufferedReader r = new java.io.BufferedReader(
                    new java.io.InputStreamReader(p.getInputStream()))) {
                String line = r.readLine();
                p.waitFor();
                if (line != null && !line.trim().isEmpty() && new File(line.trim()).isFile()) {
                    return line.trim();
                }
            }
        } catch (Exception ignore) {
            // not on PATH
        }
        return "";
    }

    /** Query the Osprey version (best effort); returns "unknown" on failure. */
    private String getOspreyVersion(String ospreyPath) {
        try {
            Process p = new ProcessBuilder(ospreyPath, "--version").redirectErrorStream(true).start();
            try (java.io.BufferedReader r = new java.io.BufferedReader(
                    new java.io.InputStreamReader(p.getInputStream()))) {
                String line = r.readLine();
                p.waitFor();
                if (line != null && !line.trim().isEmpty()) {
                    return line.trim();
                }
            }
        } catch (Exception ignore) {
            // ignore
        }
        return "unknown";
    }

    /**
     * Build the Osprey search command. The library must already contain target (and decoy)
     * spectra/RT; when a pairing manifest is supplied, Osprey runs with library-supplied decoys.
     *
     * @param msFiles  input mzML/raw files
     * @param library  spectral library (.tsv or .blib) Carafe generated
     * @param manifest optional FDRBench pairing manifest TSV (null/empty to let Osprey reverse-decoy)
     * @param outDir   output directory; the result blib is written to {@code outDir/osprey.blib}
     * @return a {@link CmdTask}, or null if no Osprey executable could be resolved
     */
    private CmdTask buildOspreyCommand(java.util.List<String> msFiles, String library, String manifest,
            String outDir, String fdrbenchOut) {
        String ospreyPath = resolveOspreyBinary();
        if (ospreyPath == null || ospreyPath.isEmpty()) {
            JOptionPane.showMessageDialog(this,
                    "Could not find an Osprey executable. Build it with scripts/build_ospreysharp.sh "
                            + "(or .bat) and place it under osprey/" + getOspreyRid()
                            + "/ next to the Carafe jar, or set its path in preferences.",
                    "Osprey Not Found", JOptionPane.WARNING_MESSAGE);
            return null;
        }
        // Make sure the binary is executable on Unix.
        File f = new File(ospreyPath);
        if (f.exists() && !System.getProperty("os.name").toLowerCase().contains("win") && !f.canExecute()) {
            f.setExecutable(true);
        }

        ArrayList<String> args = new ArrayList<>();
        args.add(ospreyPath);
        // -i is variadic in Osprey: one flag, then all input files.
        args.add("-i");
        for (String ms : msFiles) {
            args.add(ms);
        }
        args.add("-l");
        args.add(library);

        // Osprey writes its output as a SQLite .blib. SQLite cannot reliably create/lock a
        // database over a network share (UNC \\server\... or a mapped drive backed by SMB): the
        // BlibWriter does File.Delete + recreate + WAL, which fails on SMB with "unable to open
        // database file". So always have Osprey write the blib to a LOCAL temp directory and
        // move it to the requested (possibly network) outDir after the step succeeds. This also
        // speeds up the many small SQLite/WAL writes. The staging dir is keyed on a hash of outDir
        // so the train and project searches don't collide.
        String finalBlib = outDir + File.separator + "osprey.blib";
        String stageDir = new File(System.getProperty("java.io.tmpdir"),
                "carafe_osprey_blib" + File.separator + Integer.toHexString(outDir.hashCode()))
                .getAbsolutePath();
        File stageDirFile = new File(stageDir);
        stageDirFile.mkdirs();
        // Start clean so the post-step move only picks up files produced by THIS run (and so
        // BlibWriter's File.Delete + recreate runs on reliable local disk).
        File[] staleStaged = stageDirFile.listFiles();
        if (staleStaged != null) {
            for (File s : staleStaged) {
                s.delete();
            }
        }
        String outBlib = stageDir + File.separator + "osprey.blib";
        args.add("-o");
        args.add(outBlib);

        // FDRBench input TSV (requires the Osprey --fdrbench feature). Written straight to the
        // requested (possibly network) path - it is a plain TSV, not SQLite, so no local staging is
        // needed. The level follows Osprey's --fdr-level, set below from the Osprey tab.
        if (fdrbenchOut != null && !fdrbenchOut.trim().isEmpty()) {
            args.add("--fdrbench");
            args.add(fdrbenchOut);
        }

        // Library-supplied decoys via the FDRBench pairing manifest, when present.
        if (manifest != null && !manifest.trim().isEmpty()) {
            args.add("--decoys-in-library");
            args.add("--decoy-pairing-manifest");
            args.add(manifest);
        }

        // Resolution (auto/unit/hram).
        if (ospreyResolutionCombo.getSelectedItem() != null) {
            args.add("--resolution");
            args.add(ospreyResolutionCombo.getSelectedItem().toString());
        }
        // Fragment tolerance: reuse the Training Data Generation tolerance fields.
        String fragTol = fragTolField != null ? fragTolField.getText().trim() : "";
        if (!fragTol.isEmpty()) {
            args.add("--fragment-tolerance");
            args.add(fragTol);
            String unit = (fragTolUnitCombo != null && fragTolUnitCombo.getSelectedItem() != null)
                    ? fragTolUnitCombo.getSelectedItem().toString().toLowerCase()
                    : "ppm";
            args.add("--fragment-unit");
            // Osprey expects ppm|mz; map a "da"/"th" unit to "mz".
            args.add(unit.startsWith("ppm") ? "ppm" : "mz");
        }
        addOspreyOption(args, "--run-fdr", ospreyRunFdrField);
        addOspreyOption(args, "--experiment-fdr", ospreyExperimentFdrField);
        addOspreyOption(args, "--protein-fdr", ospreyProteinFdrField);
        // Osprey always uses percolator FDR.
        args.add("--fdr-method");
        args.add("percolator");
        if (ospreyFdrLevelCombo.getSelectedItem() != null) {
            args.add("--fdr-level");
            args.add(ospreyFdrLevelCombo.getSelectedItem().toString());
        }
        if (ospreySharedPeptidesCombo.getSelectedItem() != null) {
            args.add("--shared-peptides");
            args.add(ospreySharedPeptidesCombo.getSelectedItem().toString());
        }
        args.add("--threads");
        args.add(String.valueOf(Runtime.getRuntime().availableProcessors()));

        // User-supplied extra options.
        String extra = ospreyAdditionalOptionsField.getText().trim();
        if (!extra.isEmpty()) {
            Collections.addAll(args, Commandline.translateCommandline(extra));
        }

        prefs.put(PREF_OSPREY_PATH, ospreyPath);
        Cloger.getInstance().logger.info("Using Osprey " + getOspreyVersion(ospreyPath) + " at " + ospreyPath);

        CmdTask task = new CmdTask(args, "Osprey", "Running Osprey search");
        task.cmd = String.join(" ", args);
        task.input_files.addAll(msFiles);
        task.input_files.add(library);
        if (manifest != null && !manifest.trim().isEmpty()) {
            task.input_files.add(manifest);
        }
        task.out_dir = outDir;
        // Reuse is keyed on the FINAL blib (after the move), not the local staging copy.
        task.skip_check_file = finalBlib;
        // Move the locally-staged blib (and any sidecar files Osprey wrote next to it) to the
        // final output directory once the search exits cleanly.
        final String fStageDir = stageDir;
        final String fOutDir = outDir;
        final String fFinalBlib = finalBlib;
        final String fFdrbenchOut = fdrbenchOut;
        final String fManifest = manifest;
        task.postAction = () -> {
            new File(fOutDir).mkdirs();
            File[] staged = new File(fStageDir).listFiles();
            if (staged == null || staged.length == 0) {
                throw new java.io.IOException(
                        "Osprey produced no output in the staging directory: " + fStageDir);
            }
            for (File stagedFile : staged) {
                java.nio.file.Path dest = new File(fOutDir, stagedFile.getName()).toPath();
                try {
                    java.nio.file.Files.move(stagedFile.toPath(), dest,
                            java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                } catch (java.io.IOException crossDevice) {
                    // Staging dir and destination are on different volumes (the usual case for a
                    // network destination): fall back to copy + delete.
                    java.nio.file.Files.copy(stagedFile.toPath(), dest,
                            java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                    stagedFile.delete();
                }
            }
            logToConsole("[Carafe] Moved Osprey blib from local staging to " + fFinalBlib + "\n");

            // When an FDRBench input was requested, copy the pairing manifest next to it so the
            // FDRBench folder holds everything needed to run FDRBench (input TSV + pairing manifest).
            if (fFdrbenchOut != null && !fFdrbenchOut.trim().isEmpty()
                    && fManifest != null && !fManifest.trim().isEmpty()) {
                File benchDir = new File(fFdrbenchOut).getParentFile();
                File manSrc = new File(fManifest);
                if (benchDir != null && manSrc.exists()) {
                    benchDir.mkdirs();
                    java.nio.file.Files.copy(manSrc.toPath(),
                            new File(benchDir, manSrc.getName()).toPath(),
                            java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                    logToConsole("[Carafe] Copied pairing manifest into the FDRBench folder: "
                            + benchDir.getAbsolutePath() + "\n");
                }
            }
        };
        return task;
    }

    /** Append an Osprey {@code --flag value} pair when the field has a non-empty value. */
    private void addOspreyOption(ArrayList<String> args, String flag, JTextField field) {
        if (field == null) {
            return;
        }
        String v = field.getText().trim();
        if (!v.isEmpty()) {
            args.add(flag);
            args.add(v);
        }
    }

    /**
     * Map the "Initial library predictor" selection to a Koina {ms2 model, rt model} pair, or null
     * if the selection is the local Carafe predictor.
     */
    private String[] koinaModelsFor(String predictor) {
        if (predictor == null) {
            return null;
        }
        return switch (predictor) {
            case "Koina: AlphaPepDeep" -> new String[] { "AlphaPeptDeep_ms2_generic", "AlphaPeptDeep_rt_generic" };
            case "Koina: Prosit 2020 HCD" -> new String[] { "Prosit_2020_intensity_HCD", "Prosit_2019_irt" };
            case "Koina: Prosit 2020 CID" -> new String[] { "Prosit_2020_intensity_CID", "Prosit_2019_irt" };
            case "Koina: Prosit timsTOF" -> new String[] { "Prosit_2023_intensity_timsTOF", "Prosit_2019_irt" };
            case "Koina: ms2pip HCD" -> new String[] { "ms2pip_2021_HCD", "AlphaPeptDeep_rt_generic" };
            case "Koina: ms2pip timsTOF" -> new String[] { "ms2pip_timsTOF2023", "AlphaPeptDeep_rt_generic" };
            default -> null; // "Carafe (local AlphaPepDeep)"
        };
    }

    /**
     * Build the Carafe {@code -build_koina_library} command: predict an initial DIA-NN library from
     * {@code inputFasta} (the target+decoy peptide FASTA) via Koina, writing it to {@code outTsv}.
     */
    private CmdTask buildKoinaLibraryCommand(String inputFasta, String outTsv, String nceRefMzml) {
        String[] models = koinaModelsFor((String) initialLibraryPredictorCombo.getSelectedItem());
        if (models == null) {
            return null;
        }
        List<String> args = new ArrayList<>();
        String javaExec = getJavaExecutable();
        boolean exeLaunch = false;
        if (javaExec.endsWith("java.exe") || javaExec.endsWith("java")) {
            // use as is
        } else if (javaExec.endsWith("Carafe.exe")) {
            exeLaunch = true;
        } else {
            javaExec = "java";
        }
        args.add(javaExec);
        if (!exeLaunch) {
            args.add("-jar");
            args.add(getCarafeJarPath());
        }
        args.add("-build_koina_library");
        args.add(outTsv);
        args.add("-db");
        args.add(inputFasta);
        args.add("-koina_ms2_model");
        args.add(models[0]);
        args.add("-koina_rt_model");
        args.add(models[1]);
        args.add("-koina_url");
        args.add(koinaUrlField.getText().trim());
        // NCE: a number is passed through; "auto" reads the collision energy from the reference
        // mzML at runtime (Koina's intensity models for HCD/timsTOF need a collision energy).
        String nce = nceField.getText().trim();
        if (nce.equalsIgnoreCase("auto")) {
            args.add("-nce");
            args.add("auto");
            if (nceRefMzml != null && !nceRefMzml.isEmpty()) {
                args.add("-nce_ms");
                args.add(nceRefMzml);
            }
        } else if (!nce.isEmpty()) {
            args.add("-nce");
            args.add(nce);
        }
        if (msInstrumentField.getSelectedItem() != null) {
            args.add("-ms_instrument");
            args.add(msInstrumentField.getSelectedItem().toString());
        }
        args.add("-min_pep_charge");
        args.add(minPepChargeSpinner.getValue().toString());
        args.add("-max_pep_charge");
        args.add(maxPepChargeSpinner.getValue().toString());
        args.add("-min_pep_mz");
        args.add(minPepMzSpinner.getValue().toString());
        args.add("-max_pep_mz");
        args.add(maxPepMzSpinner.getValue().toString());
        args.add("-fixMod");
        args.add(fixModSelectedField.getText().trim());
        args.add("-varMod");
        args.add(varModSelectedField.getText().trim());
        args.add("-maxVar");
        args.add(maxVarSpinner.getValue().toString());
        args.add("-lf_top_n_frag");
        args.add(LibTopNFragIonsSpinner.getValue().toString());
        args.add("-lf_frag_mz_min");
        args.add(libMinFragMzSpinner.getValue().toString());

        CmdTask task = new CmdTask(args, "Carafe-Koina", "Build initial library via Koina ("
                + models[0] + ")");
        task.cmd = String.join(" ", args);
        task.input_files.add(inputFasta);
        if (nceRefMzml != null && !nceRefMzml.isEmpty()) {
            task.input_files.add(nceRefMzml);
        }
        File parent = new File(outTsv).getParentFile();
        task.out_dir = parent != null ? parent.getAbsolutePath() : ".";
        task.out_files.add(outTsv);
        task.out_files_description.add("Koina initial spectral library");
        task.skip_check_file = outTsv;
        return task;
    }

    /**
     * Build the Carafe {@code -build_entrapment_fasta} command: digest {@code inputFasta} with the
     * GUI's configured digest options into a peptide-level FASTA plus an FDRBench pairing manifest
     * (target+decoy, plus entrapment quartets when {@code withEntrapment} is true).
     *
     * <p>Entrapment peptides belong ONLY in the library-DB FASTA that feeds the finetuned library
     * used for the project search (where FDRBench measures FDP) — never in the training-DB FASTA,
     * because the training search drives AI fine-tuning and identifying random entrapment sequences
     * would pollute that training. Callers therefore pass {@code withEntrapment} explicitly rather
     * than reading the checkbox here.</p>
     */
    private CmdTask buildEntrapmentFastaCommand(String inputFasta, String outPeptideFasta, String outManifest,
            boolean withEntrapment) {
        List<String> args = new ArrayList<>();
        String javaExec = getJavaExecutable();
        boolean exeLaunch = false;
        if (javaExec.endsWith("java.exe") || javaExec.endsWith("java")) {
            // use as is
        } else if (javaExec.endsWith("Carafe.exe")) {
            exeLaunch = true;
        } else {
            javaExec = "java";
        }
        args.add(javaExec);
        if (!exeLaunch) {
            args.add("-jar");
            args.add(getCarafeJarPath());
        }
        args.add("-build_entrapment_fasta");
        args.add(outPeptideFasta);
        args.add("-db");
        args.add(inputFasta);
        args.add("-manifest");
        args.add(outManifest);
        args.add("-enzyme");
        args.add(((String) enzymeCombo.getSelectedItem()).split(":")[0]);
        args.add("-miss_c");
        args.add(missCleavageSpinner.getValue().toString());
        args.add("-minLength");
        args.add(minLengthSpinner.getValue().toString());
        args.add("-maxLength");
        args.add(maxLengthSpinner.getValue().toString());
        args.add("-min_pep_charge");
        args.add(minPepChargeSpinner.getValue().toString());
        args.add("-max_pep_charge");
        args.add(maxPepChargeSpinner.getValue().toString());
        if (withEntrapment) {
            args.add("-entrapment");
        }

        CmdTask task = new CmdTask(args, "Carafe", "Build target-decoy peptide FASTA for Osprey");
        task.cmd = String.join(" ", args);
        task.input_files.add(inputFasta);
        File parent = new File(outPeptideFasta).getParentFile();
        task.out_dir = parent != null ? parent.getAbsolutePath() : ".";
        task.out_files.add(outPeptideFasta);
        task.out_files_description.add("Peptide-level FASTA (target+decoy)");
        task.skip_check_file = outPeptideFasta;
        return task;
    }

    /**
     * Resolve a train/project MS selection into a list of mzML inputs for Osprey. Osprey
     * only reads mzML, so EVERY non-mzML acquisition (Thermo .raw, Bruker .d, ...) is routed through
     * MSConvert; an MSConvert task is queued into {@code convTasks} for those. Existing .mzML files
     * pass through unchanged.
     */
    private java.util.List<String> resolveOspreyMsInputs(java.util.List<String> selected, String fieldText,
            String convertSubDir, java.util.List<CmdTask> convTasks) {
        java.util.List<String> files = new ArrayList<>();
        java.util.List<String> toConvert = new ArrayList<>();
        java.util.List<String> candidates = new ArrayList<>();
        if (selected != null && !selected.isEmpty()) {
            candidates.addAll(selected);
        } else if (fieldText != null && !fieldText.trim().isEmpty()) {
            candidates.add(fieldText.trim());
        }
        for (String path : candidates) {
            File f = new File(path);
            String low = path.toLowerCase();
            if (low.endsWith(".mzml")) {
                files.add(path);
            } else if (low.endsWith(".d") && f.isDirectory()) {
                toConvert.add(path); // Bruker .d -> mzML for Osprey
            } else if (low.endsWith(".raw")) {
                toConvert.add(path); // Thermo .raw -> mzML for Osprey
            } else if (f.isDirectory()) {
                File[] mz = f.listFiles((d, n) -> n.toLowerCase().endsWith(".mzml"));
                if (mz != null) {
                    for (File m : mz) {
                        files.add(m.getAbsolutePath());
                    }
                }
                File[] dd = f.listFiles((d, n) -> n.toLowerCase().endsWith(".d"));
                if (dd != null) {
                    for (File m : dd) {
                        toConvert.add(m.getAbsolutePath());
                    }
                }
                File[] rw = f.listFiles((d, n) -> n.toLowerCase().endsWith(".raw"));
                if (rw != null) {
                    for (File m : rw) {
                        toConvert.add(m.getAbsolutePath());
                    }
                }
            }
        }
        if (!toConvert.isEmpty()) {
            File sd = new File(convertSubDir);
            if (!sd.exists()) {
                sd.mkdirs();
            }
            // One MSConvert task per file so they can run as separate parallel processes.
            for (String srcPath : toConvert) {
                String srcName = new File(srcPath).getName();
                String base = srcName.lastIndexOf('.') > 0
                        ? srcName.substring(0, srcName.lastIndexOf('.'))
                        : srcName;
                String mzML = convertSubDir + File.separator + base + ".mzML";
                files.add(mzML);
                String convCmd = buildMsConvertCommand(java.util.List.of(srcPath), convertSubDir);
                CmdTask conv = new CmdTask(convCmd, "MSConvert", "Convert " + srcName + " to mzML");
                conv.out_dir = convertSubDir;
                conv.input_files.add(srcPath);
                conv.skip_check_file = mzML;
                convTasks.add(conv);
            }
        }
        return files;
    }

    private void executeCommand(CmdTask command) {
        isRunning = true;
        runButton.setEnabled(false);
        stopButton.setEnabled(true);
        progressBar.setIndeterminate(true);
        progressBar.setString(command.task_description);
        timeUsageMap.clear();
        tasks.clear();

        saveParameterScreenshots();
        autoSaveRunSettings();

        if (tabbedPane != null) {
            SwingUtilities.invokeLater(() -> tabbedPane.setSelectedIndex(tabbedPane.indexOfTab("Console")));
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
        command.time_start = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
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
                        command.time_end = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
                        double minutes = (end - start) / 1e9 / 60.0;
                        command.time_used = minutes;
                        String key = "01. " + command.task_name + " - " + command.task_description;
                        timeUsageMap.put(key, minutes);
                        tasks.add(command);
                        logToConsole("\n[SUCCESS] Carafe completed successfully!\n");
                        progressBar.setString("Completed");
                        logToConsole("\n[SUMMARY] Step durations (min):\n");
                        for (java.util.Map.Entry<String, Double> e : timeUsageMap.entrySet()) {
                            //logToConsole(" - " + e.getKey() + " : " + String.format("%.2f", e.getValue()) + "\n");
                        }
                        int taskIndex = 1;
                        for (CmdTask task : tasks) {
                            logToConsole("\n[" + String.format("%02d", taskIndex++) + "] " + task.task_name + " - " + task.task_description + "\n");
                            // logToConsole("  Command: " + task.cmd + "\n");
                            logToConsole("  Output directory: " + task.out_dir + "\n");
                            if(!task.out_files.isEmpty()){
                                for(int k=0;k<task.out_files.size();k++){
                                    logToConsole("  " + task.out_files_description.get(k) + ": " + task.out_files.get(k) + "\n");
                                }
                            }
                            logToConsole("  Time start: " + task.time_start + "\n");
                            logToConsole("  Time end: " + task.time_end + "\n");
                            logToConsole("  Time used: " + String.format("%.2f", task.time_used) + " min\n");
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
        isRunning = false;
        boolean wasActive = (currentProcess != null && currentProcess.isAlive()) || !activeProcesses.isEmpty();
        terminateAllProcesses();
        if (wasActive) {
            logToConsole("\n[STOPPED] Process stopped by user.\n");
        }
        finishExecution();
    }

    /**
     * Forcibly terminate every tracked external process (MSConvert / DIA-NN / Osprey /
     * Carafe / Python) <em>and its descendants</em>. Called by Stop and by the JVM shutdown hook,
     * so closing the window mid-run does not leave orphaned converter processes behind.
     */
    private synchronized void terminateAllProcesses() {
        isRunning = false;
        java.util.List<Process> all = new java.util.ArrayList<>(activeProcesses);
        if (currentProcess != null) {
            all.add(currentProcess);
        }
        main.java.util.ProcessUtils.terminateAll(all);
        activeProcesses.clear();
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
            boolean isWindows = System.getProperty("os.name").toLowerCase().contains("windows");

            if (!path.isEmpty()) {
                File pathFile = new File(path);

                // Auto-correct: if path is a folder, look for diann executable inside
                if (pathFile.exists() && pathFile.isDirectory()) {
                    String exeName = isWindows ? "diann.exe" : "diann";
                    File diannExe = new File(pathFile, exeName);
                    if (diannExe.exists()) {
                        path = diannExe.getAbsolutePath();
                        diannPathCombo.setSelectedItem(path);
                        prefs.put(PREF_DIANN_PATH, path);
                        logToConsole("Auto-corrected folder to " + exeName + "\n");
                    }
                }
                // Auto-correct: DIA-NN.exe to diann.exe on Windows
                else if (isWindows && pathFile.getName().equalsIgnoreCase("DIA-NN.exe")) {
                    File diannExe = new File(pathFile.getParent(), "diann.exe");
                    if (diannExe.exists()) {
                        path = diannExe.getAbsolutePath();
                        diannPathCombo.setSelectedItem(path);
                        prefs.put(PREF_DIANN_PATH, path);
                        logToConsole("Auto-corrected DIA-NN.exe to diann.exe\n");
                    }
                }
            }

            if (path.isEmpty())
                errors.add("- DIA-NN executable is not specified.");
            else if (!new File(path).exists())
                errors.add("- DIA-NN executable not found.");
            else if (new File(path).isDirectory())
                errors.add("- DIA-NN path is a folder. Please specify the executable file.");
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

        // Helper to detect MS data type: "timstof", "raw", "mzml", or "unknown"
        java.util.function.Function<java.util.List<String>, String> detectMsDataType = (fileList) -> {
            boolean hasMzML = false;
            boolean hasRaw = false;
            boolean hasTimsTof = false;
            for (String p : fileList) {
                String low = p.toLowerCase();
                File f = new File(p);
                if (low.endsWith(".d") && f.isDirectory()) {
                    File analysisTdf = new File(f, "analysis.tdf");
                    if (analysisTdf.exists()) {
                        hasTimsTof = true;
                    }
                } else if (low.endsWith(".mzml")) {
                    hasMzML = true;
                } else if (low.endsWith(".raw")) {
                    hasRaw = true;
                } else if (f.isDirectory()) {
                    // Check folder contents
                    File[] dFolders = f.listFiles((d, n) -> n.toLowerCase().endsWith(".d"));
                    if (dFolders != null && dFolders.length > 0) {
                        hasTimsTof = true;
                    }
                    File[] rawFiles = f.listFiles((d, n) -> n.toLowerCase().endsWith(".raw"));
                    if (rawFiles != null && rawFiles.length > 0) {
                        hasRaw = true;
                    }
                    File[] mzmlFiles = f.listFiles((d, n) -> n.toLowerCase().endsWith(".mzml"));
                    if (mzmlFiles != null && mzmlFiles.length > 0) {
                        hasMzML = true;
                    }
                }
            }
            if (hasTimsTof && !hasRaw && !hasMzML)
                return "timstof";
            if (hasRaw && !hasTimsTof && !hasMzML)
                return "raw";
            if (hasMzML && !hasTimsTof && !hasRaw)
                return "mzml";
            if (hasTimsTof || hasRaw || hasMzML)
                return "mixed";
            return "unknown";
        };

        String trainDataType = !effectiveTrainFiles.isEmpty() ? detectMsDataType.apply(effectiveTrainFiles) : "unknown";
        String projectDataType = !effectiveProjectFiles.isEmpty() ? detectMsDataType.apply(effectiveProjectFiles)
                : "unknown";

        // Validate .d folders contain analysis.tdf
        for (String p : effectiveTrainFiles) {
            File f = new File(p);
            if (p.toLowerCase().endsWith(".d") && f.isDirectory()) {
                File analysisTdf = new File(f, "analysis.tdf");
                if (!analysisTdf.exists()) {
                    errors.add("- TIMSTOF folder missing analysis.tdf: " + f.getName());
                }
            }
        }
        for (String p : effectiveProjectFiles) {
            File f = new File(p);
            if (p.toLowerCase().endsWith(".d") && f.isDirectory()) {
                File analysisTdf = new File(f, "analysis.tdf");
                if (!analysisTdf.exists()) {
                    errors.add("- TIMSTOF folder missing analysis.tdf: " + f.getName());
                }
            }
        }

        // Cross-field consistency: Train and Project must be same type
        if (workflowIndex == 2 && !effectiveTrainFiles.isEmpty() && !effectiveProjectFiles.isEmpty()) {
            if (!trainDataType.equals("unknown") && !projectDataType.equals("unknown")
                    && !trainDataType.equals(projectDataType)) {
                errors.add("- Train MS and Project MS files must be the same type. " +
                        "Train is " + trainDataType.toUpperCase() + ", Project is " + projectDataType.toUpperCase()
                        + ".");
            }
        }

        boolean hasRaw = "raw".equals(trainDataType) || "raw".equals(projectDataType);
        boolean hasTimsTof = "timstof".equals(trainDataType) || "timstof".equals(projectDataType);

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

            case 3: // Osprey: search -> finetune -> new library
            case 4: // Osprey: end-to-end
                // 1. Train MS Files
                if (effectiveTrainFiles.isEmpty())
                    errors.add("- No Training MS data files selected.");
                else
                    for (String p : effectiveTrainFiles)
                        if (!new File(p).exists()) {
                            errors.add("- Training MS File(s) not found: " + p);
                            break;
                        }

                // 2. Train Protein DB (used to build the initial Osprey library)
                String trainDbO = trainDbFileField.getText().trim();
                if (trainDbO.isEmpty())
                    errors.add("- Training protein database (FASTA) is not specified.");
                else if (!new File(trainDbO).exists())
                    errors.add("- Training protein database file not found.");

                // 3. Library Protein DB (the new FASTA to predict after finetuning)
                String libDbO = libraryDbFileField.getText().trim();
                if (libDbO.isEmpty())
                    errors.add("- Library protein database (FASTA) is not specified.");
                else if (!new File(libDbO).exists())
                    errors.add("- Library protein database file not found.");

                // 4. Project MS Files (Workflow 5 only)
                if (workflowIndex == 4) {
                    if (effectiveProjectFiles.isEmpty())
                        errors.add("- No Project MS data files selected.");
                    else
                        for (String p : effectiveProjectFiles)
                            if (!new File(p).exists()) {
                                errors.add("- Project MS File(s) not found: " + p);
                                break;
                            }
                }

                // 5. Output Directory + 6. Python (Osprey is auto-detected/bundled; the
                // Osprey command builder reports a clear error if no executable is found).
                checkOutDir.accept(null);
                checkPython.accept(null);
                // Osprey needs mzML, so any non-mzML acquisition (.raw or Bruker .d) requires
                // MSConvert.
                if (hasRaw || hasTimsTof)
                    checkMsConvert.accept(null);
                break;
        }

        // Validate Fragment Ion Tolerance
        try {
            String tolText = fragTolField.getText().trim();
            if (tolText.isEmpty()) {
                errors.add("- Fragment Ion Mass Tolerance is empty.");
            } else {
                Double.parseDouble(tolText);
            }
        } catch (NumberFormatException e) {
            errors.add("- Fragment Ion Mass Tolerance must be a valid number.");
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

            // Skip components marked as noFreeze (e.g. Open button for output directory)
            if (component instanceof JComponent jc && Boolean.TRUE.equals(jc.getClientProperty("carafe.noFreeze"))) {
                continue;
            }

            component.setEnabled(enable);

            if (component instanceof Container) {
                enableComponents((Container) component, enable);
            }
        }
    }

    private void showHelp() {
        String helpHtml = """
                <html>
                <body style="font-family: 'Segoe UI', sans-serif; font-size: 12pt; padding: 10px;">
                <h2 style="margin-top: 0;">Carafe - AI-Powered Spectral Library Generator</h2>

                <p>Carafe generates experiment-specific in silico spectral libraries
                using deep learning for DIA data analysis.</p>

                <h3>Quick Start:</h3>
                <p>For fine-tuned library generation:</p>
                <ul style="margin-left: 20px; padding-left: 0;">
                    <li>Provide a peptide detection file (e.g., DIA-NN report.tsv or report.parquet)</li>
                    <li>Provide MS file(s) in mzML format/Thermo raw/Bruker .d format</li>
                    <li>Provide protein database (FASTA)</li>
                    <li>Configure settings and click Run</li>
                </ul>



                <p>For more information, visit:<br/>
                <a href="https://github.com/Noble-Lab/Carafe">https://github.com/Noble-Lab/Carafe</a></p>

                <h3>Citation:</h3>
                <p>Bo Wen, Chris Hsu, David Shteynberg, Wen-Feng Zeng, Michael Riffle,<br/>
                Alexis Chang, Miranda C. Mudge, Brook L. Nunn, Brendan X. MacLean,<br/>
                Matthew D. Berg, Judit Villén, Michael J. MacCoss &amp; William S. Noble.<br/>
                <a href="https://www.nature.com/articles/s41467-025-64928-4">Carafe enables high quality in silico spectral library generation
                for data-independent acquisition proteomics.</a><br/>
                <i>Nature Communications</i> 16, 9815 (2025).</p>
                </body>
                </html>
                """;

        JEditorPane editorPane = new JEditorPane("text/html", helpHtml);
        editorPane.setEditable(false);
        editorPane.setOpaque(false);
        editorPane.putClientProperty(JEditorPane.HONOR_DISPLAY_PROPERTIES, Boolean.TRUE);

        // Make hyperlinks clickable
        editorPane.addHyperlinkListener(e -> {
            if (e.getEventType() == javax.swing.event.HyperlinkEvent.EventType.ACTIVATED) {
                try {
                    java.awt.Desktop.getDesktop().browse(e.getURL().toURI());
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        });

        JScrollPane scrollPane = new JScrollPane(editorPane);
        scrollPane.setPreferredSize(new Dimension(550, 450));
        scrollPane.setBorder(null);

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

        // System.out.println("DEBUG: Using MSConvert executable: " + msConvertExec);

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

            // Consistent font size across the app
            Font defaultFont = UIManager.getFont("Label.font");
            if (defaultFont != null) {
                UIManager.put("defaultFont", defaultFont.deriveFont(13f));
            }

            // Optional: rounder components (FlatLaf supports this)
            UIManager.put("Component.arc", 12);
            UIManager.put("Button.arc", 12);
            UIManager.put("TextComponent.arc", 10);

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
                // Custom JFileChooser that navigates to a folder when a directory path is pasted
                // (except for .d folders which are valid TIMSTOF data selections)
                JFileChooser chooser = new JFileChooser() {
                    @Override
                    public void approveSelection() {
                        File[] selectedFiles = getSelectedFiles();
                        // If only one item selected and it's a directory (not a .d folder), navigate into it
                        if (selectedFiles != null && selectedFiles.length == 1) {
                            File selected = selectedFiles[0];
                            // if (selected.isDirectory() && !selected.getName().toLowerCase().endsWith(".d")) {
                            if (selected.isDirectory() && !GenericUtils.isTimsTOFData(selected.getAbsolutePath())) {
                                setCurrentDirectory(selected);
                                return; // Don't approve, just navigate
                            }
                        }
                        super.approveSelection();
                    }
                };
                String lastDir = prefs.get(PREF_LAST_DIR, System.getProperty("user.home"));
                chooser.setCurrentDirectory(new File(lastDir));
                chooser.setMultiSelectionEnabled(true);
                // Use FILES_AND_DIRECTORIES to support TIMSTOF .d folders
                chooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
                if (title != null) {
                    chooser.setDialogTitle(title);
                }
                // Custom filter that accepts files with specified extensions AND .d folders
                // (TIMSTOF)
                chooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
                    @Override
                    public boolean accept(File f) {
                        if (f.isDirectory()) {
                            // Always show directories for navigation
                            // .d folders are valid TIMSTOF data if they contain analysis.tdf
                            return true;
                        }
                        String name = f.getName().toLowerCase();
                        for (String ext : extensions) {
                            if (name.endsWith("." + ext.toLowerCase())) {
                                return true;
                            }
                        }
                        return false;
                    }

                    @Override
                    public String getDescription() {
                        return description + " or TIMSTOF .d folders";
                    }
                });
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

    // =====================================================================================
    // Parameter settings save/load (JSON). Self-contained region.
    //
    // The loader is intentionally tolerant: unknown keys are ignored and missing keys keep
    // their current/default values, so older or newer settings files still load cleanly.
    // =====================================================================================

    /** A single saveable parameter: a stable key plus get/set accessors for its component. */
    private record Setting(String key, java.util.function.Supplier<Object> getter,
            java.util.function.Consumer<Object> setter) {
    }

    private Setting textSetting(String key, JTextField f) {
        return new Setting(key, f::getText, v -> f.setText(v == null ? "" : String.valueOf(v)));
    }

    private Setting comboSetting(String key, JComboBox<String> c) {
        return new Setting(key, c::getSelectedItem, v -> {
            if (v != null) {
                c.setSelectedItem(String.valueOf(v));
            }
        });
    }

    private Setting spinnerSetting(String key, JSpinner s) {
        return new Setting(key, s::getValue, v -> applySpinnerValue(s, v));
    }

    private Setting checkSetting(String key, JCheckBox c) {
        return new Setting(key, c::isSelected, v -> c.setSelected(asBool(v)));
    }

    private void applySpinnerValue(JSpinner sp, Object v) {
        if (v == null) {
            return;
        }
        Object cur = sp.getValue();
        Number n = (v instanceof Number num) ? num : new java.math.BigDecimal(String.valueOf(v).trim());
        if (cur instanceof Integer) {
            sp.setValue(n.intValue());
        } else if (cur instanceof Long) {
            sp.setValue(n.longValue());
        } else if (cur instanceof Float) {
            sp.setValue(n.floatValue());
        } else if (cur instanceof Double) {
            sp.setValue(n.doubleValue());
        } else {
            sp.setValue(n);
        }
    }

    private boolean asBool(Object v) {
        if (v instanceof Boolean b) {
            return b;
        }
        return Boolean.parseBoolean(String.valueOf(v).trim());
    }

    /**
     * The complete list of saveable parameters.
     *
     * Note: executable paths (python/DIA-NN/MSConvert) are intentionally excluded — they are
     * machine-specific and already persisted via java Preferences.
     */
    private java.util.List<Setting> settingsRegistry() {
        java.util.List<Setting> reg = new java.util.ArrayList<>();

        // --- Workflow tab (selection + inputs) ---
        reg.add(new Setting("workflow", workflowCombo::getSelectedIndex,
                v -> workflowCombo.setSelectedIndex(((Number) v).intValue())));
        reg.add(textSetting("diann_report_file", diannReportFileField));
        reg.add(textSetting("train_ms_file", trainMsFileField));
        reg.add(textSetting("train_db_file", trainDbFileField));
        reg.add(textSetting("project_ms_file", projectMsFileField));
        reg.add(textSetting("library_db_file", libraryDbFileField));
        reg.add(textSetting("output_dir", outputDirField));
        reg.add(textSetting("carafe_additional_options", carafeAdditionalOptionsField));
        reg.add(textSetting("diann_additional_options", diannAdditionalOptionsField));
        reg.add(new Setting("train_ms_files", () -> new java.util.ArrayList<>(trainMsFiles),
                v -> setStringList(trainMsFiles, v)));
        reg.add(new Setting("project_ms_files", () -> new java.util.ArrayList<>(projectMsFiles),
                v -> setStringList(projectMsFiles, v)));

        // --- Training Data Generation tab ---
        reg.add(spinnerSetting("fdr", fdrSpinner));
        reg.add(spinnerSetting("ptm_site_prob", ptmSiteProbSpinner));
        reg.add(spinnerSetting("ptm_site_qvalue", ptmSiteQvalueSpinner));
        reg.add(textSetting("frag_tol", fragTolField));
        reg.add(comboSetting("frag_tol_unit", fragTolUnitCombo));
        reg.add(checkSetting("refine_boundary", refineBoundaryCheckbox));
        reg.add(textSetting("rt_peak_window", rtPeakWindowField));
        reg.add(spinnerSetting("xic_cor", xicCorSpinner));
        reg.add(spinnerSetting("min_frag_mz", minFragMzSpinner));
        reg.add(spinnerSetting("n_ion_min", nIonMinSpinner));
        reg.add(spinnerSetting("c_ion_min", cIonMinSpinner));

        // --- Model Training tab ---
        reg.add(comboSetting("mode", modeCombo));
        reg.add(textSetting("nce", nceField));
        reg.add(comboSetting("ms_instrument", msInstrumentField));
        reg.add(comboSetting("device", deviceCombo));

        // --- Library Generation tab ---
        reg.add(comboSetting("enzyme", enzymeCombo));
        reg.add(spinnerSetting("miss_cleavage", missCleavageSpinner));
        reg.add(textSetting("fix_mod_selected", fixModSelectedField));
        reg.add(textSetting("var_mod_selected", varModSelectedField));
        reg.add(spinnerSetting("max_var_mods", maxVarSpinner));
        reg.add(checkSetting("clip_nterm_m", clipNmCheckbox));
        reg.add(spinnerSetting("min_length", minLengthSpinner));
        reg.add(spinnerSetting("max_length", maxLengthSpinner));
        reg.add(spinnerSetting("min_pep_mz", minPepMzSpinner));
        reg.add(spinnerSetting("max_pep_mz", maxPepMzSpinner));
        reg.add(spinnerSetting("min_pep_charge", minPepChargeSpinner));
        reg.add(spinnerSetting("max_pep_charge", maxPepChargeSpinner));
        reg.add(spinnerSetting("lib_min_frag_mz", libMinFragMzSpinner));
        reg.add(spinnerSetting("lib_max_frag_mz", libMaxFragMzSpinner));
        reg.add(spinnerSetting("lib_top_n_frag_ions", LibTopNFragIonsSpinner));
        reg.add(spinnerSetting("lib_min_num_frag", libMinNumFragSpinner));
        reg.add(spinnerSetting("lib_frag_num_min", libFragNumMinSpinner));
        reg.add(comboSetting("library_format", libraryFormatCombo));
        reg.add(checkSetting("benchmark", benchmarkCheckbox));

        // Osprey settings (Workflows 4 & 5).
        reg.add(comboSetting("osprey_resolution", ospreyResolutionCombo));
        reg.add(comboSetting("osprey_fdr_level", ospreyFdrLevelCombo));
        reg.add(comboSetting("osprey_shared_peptides", ospreySharedPeptidesCombo));
        reg.add(textSetting("osprey_run_fdr", ospreyRunFdrField));
        reg.add(textSetting("osprey_experiment_fdr", ospreyExperimentFdrField));
        reg.add(textSetting("osprey_protein_fdr", ospreyProteinFdrField));
        reg.add(textSetting("osprey_additional_options", ospreyAdditionalOptionsField));
        reg.add(checkSetting("osprey_include_entrapment", includeEntrapmentCheckbox));
        reg.add(comboSetting("osprey_initial_predictor", initialLibraryPredictorCombo));
        reg.add(textSetting("koina_url", koinaUrlField));
        reg.add(textSetting("osprey_conversion_threads", ospreyConversionThreadsField));

        return reg;
    }

    private void setStringList(java.util.List<String> target, Object v) {
        target.clear();
        if (v instanceof java.util.List<?> list) {
            for (Object o : list) {
                if (o != null) {
                    target.add(String.valueOf(o));
                }
            }
        }
    }

    private void saveSettingsToFile(File file) throws IOException {
        java.util.LinkedHashMap<String, Object> map = new java.util.LinkedHashMap<>();
        map.put("_settings_version", 1);
        map.put("_carafe_version", CParameter.getVersion());
        map.put("_saved_at", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
        for (Setting s : settingsRegistry()) {
            try {
                map.put(s.key(), s.getter().get());
            } catch (Exception ex) {
                // A missing/uninitialized component should never abort the whole save.
                logToConsole("[WARN] Could not read setting '" + s.key() + "': " + ex.getMessage() + "\n");
            }
        }
        String json = JSON.toJSONString(map, JSONWriter.Feature.PrettyFormat, JSONWriter.Feature.WriteMapNullValue);
        java.nio.file.Files.writeString(file.toPath(), json, StandardCharsets.UTF_8);
    }

    private void loadSettingsFromFile(File file) throws IOException {
        String content = java.nio.file.Files.readString(file.toPath(), StandardCharsets.UTF_8);
        JSONObject obj = JSON.parseObject(content);
        if (obj == null) {
            throw new IOException("File is empty or not valid JSON.");
        }
        java.util.List<Setting> reg = settingsRegistry();
        java.util.Set<String> known = new java.util.HashSet<>();
        int applied = 0;
        for (Setting s : reg) {
            known.add(s.key());
            if (obj.containsKey(s.key())) {
                try {
                    s.setter().accept(obj.get(s.key()));
                    applied++;
                } catch (Exception ex) {
                    logToConsole("[WARN] Could not apply setting '" + s.key() + "': " + ex.getMessage() + "\n");
                }
            }
        }
        java.util.List<String> unknown = new java.util.ArrayList<>();
        for (String k : obj.keySet()) {
            if (!k.startsWith("_") && !known.contains(k)) {
                unknown.add(k);
            }
        }
        // Multi-file inputs keep only a summary ("(N files selected)") in their text field, so
        // re-render their proper state from the restored file lists. Otherwise the selection
        // shows as plain editable text and is not recognized as a real multi-file selection.
        updateFileFieldState(trainMsFileField, trainMsFiles);
        updateFileFieldState(projectMsFileField, projectMsFiles);

        logToConsole("[INFO] Loaded settings from: " + file.getAbsolutePath() + "\n");
        logToConsole("       Applied " + applied + " of " + reg.size() + " known parameters.\n");
        if (!unknown.isEmpty()) {
            logToConsole("       Ignored " + unknown.size() + " unrecognized key(s): " + String.join(", ", unknown)
                    + "\n");
        }
        // Immediately reflect the loaded values in the label indicators (skip the debounce).
        refreshAllIndicators();
    }

    /**
     * Automatically save the current parameters to {@code <outputDir>/carafe_settings.json}
     * at the start of a run, so each run leaves a reloadable record of its settings.
     * Mirrors {@link #saveParameterScreenshots()}; failures are logged, never fatal.
     */
    private void autoSaveRunSettings() {
        String outDir = outputDirField.getText().trim();
        if (outDir.isEmpty()) {
            return;
        }
        File dir = new File(outDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        File file = new File(dir, "carafe_settings.json");
        try {
            saveSettingsToFile(file);
            logToConsole("[INFO] Run settings saved to: " + file.getAbsolutePath() + "\n");
        } catch (Exception ex) {
            logToConsole("[WARN] Could not save run settings: " + ex.getMessage() + "\n");
        }
    }

    private void loadSettingsDialog() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Load Settings");
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        chooser.setFileFilter(new FileNameExtensionFilter("JSON settings (*.json)", "json"));
        chooser.setCurrentDirectory(new File(prefs.get(PREF_LAST_DIR, System.getProperty("user.home"))));
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        File file = chooser.getSelectedFile();
        if (file.getParent() != null) {
            prefs.put(PREF_LAST_DIR, file.getParent());
        }
        try {
            loadSettingsFromFile(file);
            JOptionPane.showMessageDialog(this, "Settings loaded from:\n" + file.getAbsolutePath()
                    + "\n\nSee the Console tab for details.", "Settings Loaded", JOptionPane.INFORMATION_MESSAGE);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "Failed to load settings:\n" + ex.getMessage(),
                    "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    // =====================================================================================
    // "Modified from default" label indicator.
    //
    // Each analysis parameter's label is recolored (MODIFIED_LABEL_COLOR) when its value
    // differs from the value captured at app startup, and reverts to the theme default when
    // it matches again. Typed fields are debounced (SETTING_INDICATOR_DELAY_MS) so the color
    // only updates a couple of seconds after typing stops; discrete inputs update immediately.
    //
    // Performance: purely event-driven (no polling). Label↔input association is resolved once
    // at startup from GridBagLayout constraints. A per-setting 'highlighted' flag means a label
    // is only repainted when its modified-state actually flips.
    // =====================================================================================

    private static final class TrackedSetting {
        final JLabel label;
        final JComponent input;
        final Object defaultValue;
        boolean highlighted = false;
        javax.swing.Timer debounce; // non-null only for free-text inputs

        TrackedSetting(JLabel label, JComponent input, Object defaultValue) {
            this.label = label;
            this.input = input;
            this.defaultValue = defaultValue;
        }
    }

    private final java.util.List<TrackedSetting> trackedSettings = new java.util.ArrayList<>();

    /**
     * Register every analysis parameter for the modified-from-default indicator. Called once at
     * the end of {@link #initComponents()}, when all components and their labels exist.
     */
    private void initSettingChangeIndicators() {
        JComponent[] inputs = {
                // Workflow tab — analysis options only (file/path/executable inputs are excluded)
                carafeAdditionalOptionsField, diannAdditionalOptionsField,
                // Training Data Generation
                fdrSpinner, ptmSiteProbSpinner, ptmSiteQvalueSpinner, fragTolField, fragTolUnitCombo,
                refineBoundaryCheckbox, rtPeakWindowField, xicCorSpinner,
                minFragMzSpinner, nIonMinSpinner, cIonMinSpinner,
                // Model Training
                modeCombo, nceField, msInstrumentField, deviceCombo,
                // Library Generation
                enzymeCombo, missCleavageSpinner, fixModSelectedField, varModSelectedField, maxVarSpinner,
                clipNmCheckbox, minLengthSpinner, maxLengthSpinner, minPepMzSpinner, maxPepMzSpinner,
                minPepChargeSpinner, maxPepChargeSpinner, libMinFragMzSpinner, libMaxFragMzSpinner,
                LibTopNFragIonsSpinner, libMinNumFragSpinner, libFragNumMinSpinner, libraryFormatCombo,
                benchmarkCheckbox,
                // Osprey
                ospreyResolutionCombo, ospreyFdrLevelCombo, ospreySharedPeptidesCombo,
                ospreyRunFdrField, ospreyExperimentFdrField, ospreyProteinFdrField,
                ospreyAdditionalOptionsField, includeEntrapmentCheckbox, initialLibraryPredictorCombo,
                ospreyConversionThreadsField
        };
        for (JComponent input : inputs) {
            if (input == null) {
                continue;
            }
            JLabel label = findLabelFor(input);
            if (label == null) {
                continue; // no adjacent label found — skip silently
            }
            TrackedSetting ts = new TrackedSetting(label, input, indicatorValue(input));
            trackedSettings.add(ts);
            attachIndicatorListener(ts);
        }
    }

    /**
     * Find the label that sits immediately to the left of {@code input} (same row, previous
     * column) within a GridBagLayout. Returns null if the layout/association can't be resolved.
     */
    private JLabel findLabelFor(JComponent input) {
        Container parent = input.getParent();
        if (parent == null || !(parent.getLayout() instanceof GridBagLayout gbl)) {
            return null;
        }
        GridBagConstraints ic = gbl.getConstraints(input);
        if (ic.gridx == GridBagConstraints.RELATIVE || ic.gridy == GridBagConstraints.RELATIVE) {
            return null;
        }
        int wantX = ic.gridx - 1;
        int wantY = ic.gridy;
        for (Component c : parent.getComponents()) {
            if (c instanceof JLabel lbl) {
                GridBagConstraints lc = gbl.getConstraints(lbl);
                if (lc.gridy == wantY && lc.gridx == wantX) {
                    return lbl;
                }
            }
        }
        return null;
    }

    /** Current comparable value of an input, used for both the default snapshot and comparisons. */
    private Object indicatorValue(JComponent c) {
        if (c instanceof JTextField tf) {
            return tf.getText().trim();
        }
        if (c instanceof JSpinner sp) {
            return sp.getValue();
        }
        if (c instanceof JComboBox<?> cb) {
            return cb.getSelectedItem();
        }
        if (c instanceof JCheckBox ck) {
            return ck.isSelected();
        }
        return null;
    }

    private void attachIndicatorListener(TrackedSetting ts) {
        JComponent c = ts.input;
        if (c instanceof JTextField tf) {
            // Typed input: debounce, plus an immediate refresh when focus leaves the field.
            ts.debounce = new javax.swing.Timer(SETTING_INDICATOR_DELAY_MS, e -> refreshIndicator(ts));
            ts.debounce.setRepeats(false);
            tf.getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
                @Override
                public void insertUpdate(javax.swing.event.DocumentEvent e) {
                    ts.debounce.restart();
                }

                @Override
                public void removeUpdate(javax.swing.event.DocumentEvent e) {
                    ts.debounce.restart();
                }

                @Override
                public void changedUpdate(javax.swing.event.DocumentEvent e) {
                    ts.debounce.restart();
                }
            });
            tf.addFocusListener(new java.awt.event.FocusAdapter() {
                @Override
                public void focusLost(java.awt.event.FocusEvent e) {
                    ts.debounce.stop();
                    refreshIndicator(ts);
                }
            });
        } else if (c instanceof JSpinner sp) {
            sp.addChangeListener(e -> refreshIndicator(ts));
        } else if (c instanceof JComboBox<?> cb) {
            cb.addActionListener(e -> refreshIndicator(ts));
        } else if (c instanceof JCheckBox ck) {
            ck.addActionListener(e -> refreshIndicator(ts));
        }
    }

    /**
     * Compare a current value to its default. Numeric values (e.g. spinner doubles) are compared
     * with a small relative tolerance so that floating-point drift from arrow-button arithmetic
     * (0.01 + 0.005 - 0.005 != 0.01) is not mistaken for a real change.
     */
    private boolean valuesEqual(Object a, Object b) {
        if (a instanceof Number na && b instanceof Number nb) {
            double da = na.doubleValue();
            double db = nb.doubleValue();
            double tol = 1e-9 * Math.max(1.0, Math.max(Math.abs(da), Math.abs(db)));
            return Math.abs(da - db) <= tol;
        }
        return java.util.Objects.equals(a, b);
    }

    /** Recolor a single label only if its modified-state changed (avoids needless repaints). */
    private void refreshIndicator(TrackedSetting ts) {
        boolean modified = !valuesEqual(indicatorValue(ts.input), ts.defaultValue);
        if (modified == ts.highlighted) {
            return;
        }
        ts.highlighted = modified;
        // null restores the look-and-feel default, so theme switches are handled automatically.
        ts.label.setForeground(modified ? MODIFIED_LABEL_COLOR : null);
    }

    /** Re-evaluate every tracked label now (used after settings are loaded programmatically). */
    private void refreshAllIndicators() {
        for (TrackedSetting ts : trackedSettings) {
            if (ts.debounce != null) {
                ts.debounce.stop();
            }
            refreshIndicator(ts);
        }
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
