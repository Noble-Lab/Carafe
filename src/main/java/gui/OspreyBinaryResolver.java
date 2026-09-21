package main.java.gui;

/**
 * Precedence rules for locating the Osprey executable, factored out of {@link CarafeGUI} so the
 * ordering can be unit-tested without constructing the Swing UI.
 */
public final class OspreyBinaryResolver {

    private OspreyBinaryResolver() {
    }

    /**
     * Pick the Osprey executable by precedence: a build bundled with the installer wins, then a
     * user-saved path (source / command-line runs), then {@code ~/.carafe/osprey}, then one found
     * on the system PATH. Each argument is an already-existence-checked absolute path, or
     * {@code null}/blank if that source did not resolve. Returns {@code ""} when nothing resolved.
     *
     * <p>Bundled-before-saved is deliberate: an MSI install always ships a matched, tested Osprey,
     * and it must not be shadowed by a stale explicit path left over from a source/dev build. The
     * saved path still wins for source/command-line runs, where no Osprey is bundled next to the
     * jar so {@code bundled} is {@code null}.
     */
    public static String choose(String bundled, String saved, String home, String onPath) {
        for (String candidate : new String[] { bundled, saved, home, onPath }) {
            if (candidate != null && !candidate.trim().isEmpty()) {
                return candidate;
            }
        }
        return "";
    }
}
