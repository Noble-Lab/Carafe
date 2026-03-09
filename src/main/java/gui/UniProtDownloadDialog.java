package main.java.gui;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Dialog for downloading protein databases from UniProt.
 */
public class UniProtDownloadDialog extends JDialog {

    // Organism options with UniProt proteome IDs
    private static final String[][] ORGANISMS = {
            { "Homo sapiens (Human)", "UP000005640" },
            { "Mus musculus (Mouse)", "UP000000589" },
            { "Saccharomyces cerevisiae (Yeast)", "UP000002311" },
            { "Escherichia coli (K-12)", "UP000000625" },
            { "Arabidopsis thaliana", "UP000006548" }
    };

    private final JTextField targetField;
    private final JTextField outputDirField;

    private ButtonGroup organismGroup;
    private JRadioButton[] organismButtons;
    private JRadioButton otherButton;
    private JTextField otherProteomeField;

    private JCheckBox reviewedCheckbox;
    private JCheckBox isoformsCheckbox;
    private JCheckBox contaminantsCheckbox;

    private JTextField spikeInField;
    private JLabel spikeInLabel;
    private JButton spikeInBrowseButton;
    private JTextField downloadDirField;
    private JButton downloadDirBrowseButton;
    private JButton downloadButton;
    private JButton cancelButton;
    private JProgressBar progressBar;
    private JLabel statusLabel;

    private SwingWorker<Integer, String> downloadWorker;

    private static final String contaminants_protein_file = "contaminants.fasta";

    public UniProtDownloadDialog(Frame owner, JTextField targetField, JTextField outputDirField) {
        super(owner, "Download Protein Database", true);
        this.targetField = targetField;
        this.outputDirField = outputDirField;
        // Remove icon from dialog title bar using FlatLaf property
        getRootPane().putClientProperty("JRootPane.titleBarShowIcon", false);
        initComponents();
        pack();
        setLocationRelativeTo(owner);
    }

    private void initComponents() {
        setLayout(new BorderLayout(10, 10));
        ((JPanel) getContentPane()).setBorder(BorderFactory.createEmptyBorder(15, 15, 15, 15));

        // Main content panel
        JPanel contentPanel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.weightx = 1.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(0, 0, 10, 0);

        // Organism selection panel
        JPanel organismPanel = new JPanel();
        organismPanel.setLayout(new BoxLayout(organismPanel, BoxLayout.Y_AXIS));
        organismPanel.setBorder(createRoundedTitledBorder("Select organism / Input proteome ID"));

        organismGroup = new ButtonGroup();
        organismButtons = new JRadioButton[ORGANISMS.length];

        for (int i = 0; i < ORGANISMS.length; i++) {
            organismButtons[i] = new JRadioButton(ORGANISMS[i][0] + " - " + ORGANISMS[i][1]);
            organismButtons[i].setActionCommand(ORGANISMS[i][1]);
            organismButtons[i].setAlignmentX(Component.LEFT_ALIGNMENT);
            organismGroup.add(organismButtons[i]);
            organismPanel.add(organismButtons[i]);
            if (i == 0)
                organismButtons[i].setSelected(true);
        }

        // Other option with text field - remove horizontal gap to align with radio
        // buttons above
        JPanel otherPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        otherPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
        otherButton = new JRadioButton("Other:");
        otherButton.setActionCommand("OTHER");
        organismGroup.add(otherButton);
        otherProteomeField = new JTextField(20);
        otherProteomeField.setToolTipText("Enter UniProt proteome ID (e.g., UP000000XXX)");
        otherProteomeField.putClientProperty("JTextField.placeholderText", "e.g., UP000005640");
        otherProteomeField.setEnabled(false);
        otherButton.addActionListener(e -> otherProteomeField.setEnabled(otherButton.isSelected()));
        for (JRadioButton btn : organismButtons) {
            btn.addActionListener(e -> otherProteomeField.setEnabled(false));
        }
        otherPanel.add(otherButton);
        otherPanel.add(otherProteomeField);
        organismPanel.add(otherPanel);

        contentPanel.add(organismPanel, gbc);

        // Options panel
        gbc.gridy++;
        JPanel optionsPanel = new JPanel();
        optionsPanel.setLayout(new BoxLayout(optionsPanel, BoxLayout.Y_AXIS));
        optionsPanel.setBorder(createRoundedTitledBorder("Options"));

        reviewedCheckbox = new JCheckBox("Reviewed sequences only (Swiss-Prot)", true);
        reviewedCheckbox.setToolTipText("Download only manually reviewed entries from Swiss-Prot");

        isoformsCheckbox = new JCheckBox("Include isoforms", false);
        isoformsCheckbox.setToolTipText("Include alternative protein isoforms");

        contaminantsCheckbox = new JCheckBox("Add common contaminants (cRAP)", true);
        contaminantsCheckbox.setToolTipText("Append common laboratory contaminants to the database");

        optionsPanel.add(reviewedCheckbox);
        optionsPanel.add(isoformsCheckbox);
        optionsPanel.add(contaminantsCheckbox);

        contentPanel.add(optionsPanel, gbc);

        // Spike-in sequences panel
        gbc.gridy++;
        JPanel spikeInPanel = new JPanel(new BorderLayout(5, 0));
        spikeInPanel.setBorder(createRoundedTitledBorder("Spike-in sequences"));

        JPanel spikeInRow = new JPanel(new BorderLayout(5, 0));
        spikeInLabel = new JLabel("FASTA file path");
        spikeInField = new JTextField();
        spikeInField.setToolTipText("Optional: Path to spike-in FASTA file (e.g., iRT standards)");
        spikeInField.putClientProperty("JTextField.placeholderText", "Optional");
        spikeInBrowseButton = new JButton("Browse");
        spikeInBrowseButton.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser();
            chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
            chooser.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("FASTA Files", "fasta", "fa"));
            if (!spikeInField.getText().isEmpty()) {
                chooser.setCurrentDirectory(new File(spikeInField.getText()).getParentFile());
            } else if (!downloadDirField.getText().isEmpty()) {
                chooser.setCurrentDirectory(new File(downloadDirField.getText()));
            }
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                spikeInField.setText(chooser.getSelectedFile().getAbsolutePath());
            }
        });
        spikeInRow.add(spikeInLabel, BorderLayout.WEST);
        spikeInRow.add(spikeInField, BorderLayout.CENTER);
        spikeInRow.add(spikeInBrowseButton, BorderLayout.EAST);
        spikeInPanel.add(spikeInRow, BorderLayout.CENTER);

        contentPanel.add(spikeInPanel, gbc);

        // Download directory panel
        gbc.gridy++;
        JPanel dirPanel = new JPanel(new BorderLayout(5, 0));
        dirPanel.setBorder(createRoundedTitledBorder("Download to"));

        downloadDirField = new JTextField();
        // Default to output directory if available
        String defaultDir = (outputDirField != null) ? outputDirField.getText().trim() : "";
        downloadDirField.setText(defaultDir);

        downloadDirBrowseButton = new JButton("Browse");
        downloadDirBrowseButton.setToolTipText(
                "When the Output Directory from the main window is set, the download directory here could be empty.\n" +
                        "The database file will be downloaded to the Output Directory.");
        downloadDirBrowseButton.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser();
            chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (!downloadDirField.getText().isEmpty()) {
                chooser.setCurrentDirectory(new File(downloadDirField.getText()));
            }
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                downloadDirField.setText(chooser.getSelectedFile().getAbsolutePath());
            }
        });

        dirPanel.add(downloadDirField, BorderLayout.CENTER);
        dirPanel.add(downloadDirBrowseButton, BorderLayout.EAST);

        contentPanel.add(dirPanel, gbc);

        // Progress panel
        gbc.gridy++;
        gbc.insets = new Insets(0, 0, 0, 0); // No bottom margin for last item
        JPanel progressPanel = new JPanel(new BorderLayout(5, 5));
        progressBar = new JProgressBar();
        progressBar.setIndeterminate(false);
        progressBar.setStringPainted(true);
        progressBar.setString("Ready");
        progressBar.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        progressBar.setBorder(BorderFactory.createEmptyBorder(6, 0, 6, 0));
        statusLabel = new JLabel(" ");
        progressPanel.add(progressBar, BorderLayout.CENTER);
        progressPanel.add(statusLabel, BorderLayout.SOUTH);

        contentPanel.add(progressPanel, gbc);

        add(contentPanel, BorderLayout.CENTER);

        // Button panel
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 0));
        downloadButton = new JButton("Download");
        // Make primary button visually primary (FlatLaf supports this)
        downloadButton.putClientProperty("JButton.buttonType", "default");
        downloadButton.addActionListener(this::onDownload);
        cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(e -> onCancel());
        buttonPanel.add(cancelButton);
        buttonPanel.add(downloadButton);

        add(buttonPanel, BorderLayout.SOUTH);

        setMinimumSize(new Dimension(450, 400));
    }

    private void onDownload(ActionEvent e) {
        // Validate download directory
        String downloadDir = downloadDirField.getText().trim();
        if (downloadDir.isEmpty()) {
            JOptionPane.showMessageDialog(this,
                    "Please specify a download directory.",
                    "Download Directory Required", JOptionPane.WARNING_MESSAGE);
            return;
        }

        File dir = new File(downloadDir);
        if (!dir.exists()) {
            int result = JOptionPane.showConfirmDialog(this,
                    "Directory does not exist. Create it?",
                    "Create Directory", JOptionPane.YES_NO_OPTION);
            if (result == JOptionPane.YES_OPTION) {
                if (!dir.mkdirs()) {
                    JOptionPane.showMessageDialog(this,
                            "Failed to create directory.",
                            "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
            } else {
                return;
            }
        }

        // Get proteome ID
        String proteomeId;
        String organismName;
        if (otherButton.isSelected()) {
            proteomeId = otherProteomeField.getText().trim();
            if (proteomeId.isEmpty()) {
                JOptionPane.showMessageDialog(this,
                        "Please enter a proteome ID.",
                        "Proteome ID Required", JOptionPane.WARNING_MESSAGE);
                return;
            }
            organismName = "custom";
        } else {
            proteomeId = organismGroup.getSelection().getActionCommand();
            // Find organism name
            organismName = "unknown";
            for (int i = 0; i < ORGANISMS.length; i++) {
                if (ORGANISMS[i][1].equals(proteomeId)) {
                    organismName = ORGANISMS[i][0].split(" \\(")[1].replace(")", "").toLowerCase();
                    break;
                }
            }
        }

        // Build URL
        String url = buildUniProtUrl(proteomeId);

        // Generate filename
        String filename = generateFilename(organismName);
        File outputFile = new File(dir, filename);

        // Disable controls during download
        setControlsEnabled(false);
        progressBar.setIndeterminate(true);
        progressBar.setString("Connecting...");
        statusLabel.setText("Connecting to UniProt...");

        // Start download
        final String finalOrganismName = organismName;
        downloadWorker = new SwingWorker<Integer, String>() {
            @Override
            protected Integer doInBackground() throws Exception {
                publish("Downloading from UniProt...");

                URL urlObj = new URL(url);
                HttpURLConnection conn = (HttpURLConnection) urlObj.openConnection();
                conn.setRequestProperty("Accept", "text/plain");
                conn.setConnectTimeout(30000);
                conn.setReadTimeout(300000); // 5 minutes for large databases

                int responseCode = conn.getResponseCode();
                if (responseCode != 200) {
                    throw new IOException("UniProt returned error: " + responseCode);
                }

                int sequenceCount = 0;
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8));
                        BufferedWriter writer = new BufferedWriter(
                                new OutputStreamWriter(new FileOutputStream(outputFile), StandardCharsets.UTF_8))) {

                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (isCancelled())
                            break;
                        writer.write(line);
                        writer.newLine();
                        if (line.startsWith(">")) {
                            sequenceCount++;
                            if (sequenceCount % 1000 == 0) {
                                publish("Downloaded " + sequenceCount + " sequences...");
                            }
                        }
                    }
                    publish("Downloaded " + sequenceCount + " sequences from UniProt.");

                    // Append contaminants if requested
                    if (contaminantsCheckbox.isSelected()) {
                        publish("Appending contaminants...");
                        appendContaminants(writer);
                    }

                    // Append spike-in sequences if provided
                    String spikeInPath = spikeInField.getText().trim();
                    if (!spikeInPath.isEmpty()) {
                        File spikeInFile = new File(spikeInPath);
                        if (spikeInFile.exists()) {
                            publish("Appending spike-in sequences...");
                            appendSpikeIn(writer, spikeInFile);
                        }
                    }
                }

                return sequenceCount;
            }

            @Override
            protected void process(java.util.List<String> chunks) {
                if (!chunks.isEmpty()) {
                    statusLabel.setText(chunks.get(chunks.size() - 1));
                }
            }

            @Override
            protected void done() {
                progressBar.setIndeterminate(false);
                setControlsEnabled(true);

                try {
                    Integer count = get(); // Get the sequence count
                    statusLabel.setText("Download complete! (" + count + " proteins)");
                    progressBar.setString("Complete (" + count + " proteins)");
                    progressBar.setValue(100);

                    // Set the target field to the downloaded file
                    targetField.setText(outputFile.getAbsolutePath());

                    JOptionPane.showMessageDialog(UniProtDownloadDialog.this,
                            "Downloaded " + count + " proteins to:\n" + outputFile.getAbsolutePath(),
                            "Download Complete", JOptionPane.INFORMATION_MESSAGE);

                    dispose();
                } catch (Exception ex) {
                    statusLabel.setText("Download failed.");
                    progressBar.setString("Failed");
                    JOptionPane.showMessageDialog(UniProtDownloadDialog.this,
                            "Download failed: " + ex.getMessage(),
                            "Error", JOptionPane.ERROR_MESSAGE);
                    // Clean up partial file
                    if (outputFile.exists()) {
                        outputFile.delete();
                    }
                }
            }
        };

        downloadWorker.execute();
    }

    private String buildUniProtUrl(String proteomeId) {
        StringBuilder url = new StringBuilder();
        url.append("https://rest.uniprot.org/uniprotkb/stream?");
        url.append("query=proteome:").append(proteomeId);

        if (reviewedCheckbox.isSelected()) {
            url.append("+AND+reviewed:true");
        }

        url.append("&format=fasta");

        if (isoformsCheckbox.isSelected()) {
            url.append("&includeIsoform=true");
        }

        return url.toString();
    }

    private String generateFilename(String organismName) {
        StringBuilder name = new StringBuilder();
        name.append(organismName.replaceAll("[^a-zA-Z0-9]", "_"));
        name.append("_");
        name.append(reviewedCheckbox.isSelected() ? "reviewed" : "all");
        name.append("_");

        // Add timestamp: date_HH-mm-ss
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
        name.append(now.format(formatter));

        if (contaminantsCheckbox.isSelected()) {
            name.append("_contaminants");
        }

        if (isoformsCheckbox.isSelected()) {
            name.append("_isoforms");
        }

        name.append(".fasta");
        return name.toString();
    }

    private void appendContaminants(BufferedWriter writer) throws IOException {
        // Try to load bundled contaminants from resources
        try (InputStream is = getClass().getResourceAsStream("/" + contaminants_protein_file)) {
            if (is != null) {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(is, StandardCharsets.UTF_8))) {
                    writer.newLine();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        writer.write(line);
                        writer.newLine();
                    }
                }
            } else {
                // Contaminants file not bundled - skip silently
                System.err.println("Warning: " + contaminants_protein_file + " not found in resources");
            }
        }
    }

    private void appendSpikeIn(BufferedWriter writer, File spikeInFile) throws IOException {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(spikeInFile), StandardCharsets.UTF_8))) {
            writer.newLine();
            writer.write("# Spike-in sequences");
            writer.newLine();
            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line);
                writer.newLine();
            }
        }
    }

    private void setControlsEnabled(boolean enabled) {
        downloadButton.setEnabled(enabled);
        for (JRadioButton btn : organismButtons) {
            btn.setEnabled(enabled);
        }
        otherButton.setEnabled(enabled);
        otherProteomeField.setEnabled(enabled && otherButton.isSelected());
        reviewedCheckbox.setEnabled(enabled);
        isoformsCheckbox.setEnabled(enabled);
        contaminantsCheckbox.setEnabled(enabled);
        spikeInLabel.setEnabled(enabled);
        spikeInField.setEnabled(enabled);
        spikeInBrowseButton.setEnabled(enabled);
        downloadDirField.setEnabled(enabled);
        downloadDirBrowseButton.setEnabled(enabled);
    }

    private void onCancel() {
        if (downloadWorker != null && !downloadWorker.isDone()) {
            downloadWorker.cancel(true);
        }
        dispose();
    }

    public void showDialog() {
        setVisible(true);
    }

    private TitledBorder createRoundedTitledBorder(String title) {
        // Separator-line style: horizontal line with title, no box border
        return BorderFactory.createTitledBorder(
                BorderFactory.createMatteBorder(1, 0, 0, 0, Color.LIGHT_GRAY),
                title,
                TitledBorder.CENTER,
                TitledBorder.TOP);
    }
}
