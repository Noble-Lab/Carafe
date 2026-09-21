package main.java.db;

import main.java.util.Cloger;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Integrity checks on the entrapment pairing, so a broken pairing is refused at the source rather
 * than worked around downstream.
 *
 * <p>A paired FDP estimator (FDRBench, and Osprey's entrapment diagnostics) assumes every
 * entrapment peptide has a target twin. When that assumption fails the estimator either crashes or
 * silently mis-scales, and the failure surfaces hours later in a search rather than seconds after
 * library generation. Osprey grew a whole reconciliation class to absorb exactly this; these
 * checks exist so it does not have to.</p>
 *
 * <p>There are two distinct failure surfaces and they need separate checks:</p>
 * <ol>
 *   <li><b>Manifest integrity</b> ({@link #validateQuartets}) - what the generator emitted. Cheap,
 *       runs at write time, catches generator bugs.</li>
 *   <li><b>Library against manifest</b> ({@link #validateLibraryAgainstManifest}) - what the
 *       predictor produced from it. This is the one that matters most, because the library is
 *       allowed to contain peptide FORMS the manifest never had, and a form that lands on only one
 *       side of a pair is precisely the defect that is invisible to a manifest-only check.</li>
 * </ol>
 */
public final class EntrapmentPairingValidator {

    private EntrapmentPairingValidator() {
    }

    /** A single integrity violation: what was wrong, and enough examples to act on it. */
    public static final class Violation {
        public final String kind;
        public final int count;
        public final List<String> examples;

        Violation(String kind, int count, List<String> examples) {
            this.kind = kind;
            this.count = count;
            this.examples = examples;
        }

        @Override
        public String toString() {
            return kind + ": " + count + (examples.isEmpty() ? "" : " (e.g. " + String.join(", ", examples) + ")");
        }
    }

    private static final int MAX_EXAMPLES = 5;

    /**
     * Validate the quartets the generator is about to write.
     *
     * <p>Checks that every quartet has a target, that a quartet SELECTED to carry entrapment
     * actually has one (a bare target+decoy pair emitted under an entrapment design silently
     * changes the entrapment ratio), and that no sequence is claimed by two different pair
     * indices - which would make the pairing ambiguous and the FDP denominator wrong.</p>
     */
    public static List<Violation> validateQuartets(List<EntrapmentFastaGear.Quartet> quartets) {
        // Counts are tracked separately from examples: the example lists are capped, so using
        // their size as the count would under-report any violation with more than MAX_EXAMPLES
        // instances - which is exactly the case where the number matters.
        Counter missingTarget = new Counter();
        Counter selectedWithoutEntrapment = new Counter();
        Counter entrapmentEqualsTarget = new Counter();
        Set<String> targetSeqs = new HashSet<>();
        Map<String, Integer> generatedToPair = new HashMap<>();

        int pairIdx = 0;
        for (EntrapmentFastaGear.Quartet q : quartets) {
            if (q.target == null || q.target.isEmpty()) {
                missingTarget.hit("pair_index " + pairIdx);
            }
            if (q.entrapmentSelected && q.pTarget == null) {
                selectedWithoutEntrapment.hit(q.target);
            }
            if (q.pTarget != null && q.pTarget.equals(q.target)) {
                entrapmentEqualsTarget.hit(q.target);
            }
            // Only a TARGET colliding with a generated sequence is a defect. Generated sequences
            // colliding with each other across pairs is normal and unavoidable: with 1.4M short
            // tryptic peptides, two different shuffles landing on the same string happens
            // thousands of times, and the generator only ever promised that an entrapment does not
            // collide with a REAL TARGET (that is what its collision-drop pass enforces). An
            // earlier version of this check asserted global uniqueness and rejected every real
            // library at 526 collisions.
            if (q.target != null) {
                targetSeqs.add(q.target);
            }
            for (String seq : new String[] { q.pTarget, q.decoy, q.pDecoy }) {
                if (seq != null) {
                    generatedToPair.putIfAbsent(seq, pairIdx);
                }
            }
            pairIdx++;
        }

        // A generated sequence equal to some REAL target is the collision the generator's
        // collision-drop pass exists to prevent, so one surviving here is a genuine generator bug
        // and not a statistical inevitability.
        Counter collidesWithRealTarget = new Counter();
        for (Map.Entry<String, Integer> e : generatedToPair.entrySet()) {
            if (targetSeqs.contains(e.getKey())) {
                collidesWithRealTarget.hit(e.getKey());
            }
        }

        List<Violation> out = new ArrayList<>();
        missingTarget.addTo(out, "quartet with no target");
        selectedWithoutEntrapment.addTo(out, "entrapment-selected quartet with no entrapment");
        entrapmentEqualsTarget.addTo(out, "entrapment sequence equal to its target");
        collidesWithRealTarget.addTo(out, "generated sequence equal to a real target");
        return out;
    }

    /**
     * Validate a predicted library against the manifest that generated it.
     *
     * <p>The library may legitimately contain forms the manifest lacks. What it may NOT contain is
     * an entrapment peptide with no target twin: a paired estimator cannot consume that row. The
     * complementary case - a target with no entrapment twin - is not fatal, but it is a
     * subpopulation whose entrapment ratio is zero, so it silently biases any entrapment-derived
     * FDP downward and is reported too.</p>
     *
     * @param libraryEntrapment entrapment (p_target) sequences present in the predicted library
     * @param libraryTargets    target sequences present in the predicted library
     * @param manifestPairs     manifest sequence -> pair index, for every type
     * @param manifestTypes     manifest sequence -> peptide_type (target / p_target / ...)
     */
    public static List<Violation> validateLibraryAgainstManifest(
            Collection<String> libraryEntrapment,
            Collection<String> libraryTargets,
            Map<String, Integer> manifestPairs,
            Map<String, String> manifestTypes) {

        Map<Integer, String> targetOfPair = new HashMap<>();
        Map<Integer, String> entrapmentOfPair = new HashMap<>();
        for (Map.Entry<String, Integer> e : manifestPairs.entrySet()) {
            String type = manifestTypes.get(e.getKey());
            if ("target".equals(type)) {
                targetOfPair.put(e.getValue(), e.getKey());
            } else if ("p_target".equals(type)) {
                entrapmentOfPair.put(e.getValue(), e.getKey());
            }
        }
        Set<String> libTargetSet = new HashSet<>(libraryTargets);
        Set<String> libEntrapmentSet = new HashSet<>(libraryEntrapment);

        // A CLIPPED entrapment does not pair with its manifest target - it pairs with the CLIP of
        // that target, which can only exist when the target itself starts with M. Checking merely
        // that the pair has some library target is not enough, and was the first form of this
        // method: it accepted exactly the sequences that shipped broken.
        Counter orphanEntrapment = new Counter();
        for (String e : libraryEntrapment) {
            Integer idx = manifestPairs.get(e);
            String requiredTarget;
            if (idx != null) {
                requiredTarget = targetOfPair.get(idx);
            } else {
                Integer clippedIdx = manifestPairs.get("M" + e);
                if (clippedIdx == null) {
                    // Not a manifest peptide and not the Met-clip of one: an unexplained extra,
                    // equally unusable by a paired estimator.
                    orphanEntrapment.hit(e);
                    continue;
                }
                String t = targetOfPair.get(clippedIdx);
                requiredTarget = (t != null && t.startsWith("M")) ? t.substring(1) : null;
            }
            if (requiredTarget == null || !libTargetSet.contains(requiredTarget)) {
                orphanEntrapment.hit(e);
            }
        }

        // Only pairs the manifest actually gave an entrapment can be "uncovered". At an entrapment
        // ratio below 1.0 most targets deliberately have none, and flagging those would bury the
        // real signal under one violation per unentrapped target - so absence in the MANIFEST means
        // by design, and only absence in the LIBRARY of an entrapment the manifest does list is a
        // defect.
        Counter uncoveredTargets = new Counter();
        for (String t : libraryTargets) {
            Integer idx = manifestPairs.get(t);
            String manifestEntrapment;
            boolean clipped = false;
            if (idx != null) {
                manifestEntrapment = entrapmentOfPair.get(idx);
            } else {
                Integer clippedIdx = manifestPairs.get("M" + t);
                if (clippedIdx == null) {
                    continue;
                }
                manifestEntrapment = entrapmentOfPair.get(clippedIdx);
                clipped = true;
            }
            if (manifestEntrapment == null) {
                continue; // by design: this pair was never selected to carry entrapment
            }
            String required = manifestEntrapment;
            if (clipped) {
                required = manifestEntrapment.startsWith("M") ? manifestEntrapment.substring(1) : null;
            }
            if (required == null || !libEntrapmentSet.contains(required)) {
                uncoveredTargets.hit(t);
            }
        }

        List<Violation> out = new ArrayList<>();
        orphanEntrapment.addTo(out, "entrapment peptide with no target twin");
        uncoveredTargets.addTo(out, "target with no entrapment coverage");
        return out;
    }

    /**
     * Apply the project's policy to a set of violations: refuse by default, warn when the caller
     * has explicitly opted out.
     *
     * <p>Separated from the caller so the decision itself is testable without having to induce a
     * real generator defect - which, by design, is not something the generator can be made to do.
     * The escape hatch matters because these invariants are asserted against small fixtures but
     * only exercised for real at 1.4M quartets; without it, a check that turns out to be wrong
     * about what the generator guarantees leaves no way to produce a manifest at all.</p>
     *
     * @param violations       result of {@link #validateQuartets}; nothing happens when empty
     * @param failOnViolation  true to throw, false to log a warning and continue
     * @throws IllegalStateException when {@code failOnViolation} and {@code violations} is non-empty
     */
    public static void enforce(List<Violation> violations, boolean failOnViolation) {
        if (violations.isEmpty()) {
            return;
        }
        String detail = describe(violations);
        if (failOnViolation) {
            throw new IllegalStateException(
                    "Entrapment manifest failed integrity checks: " + detail
                    + ". These pairings would break a paired FDP estimator (FDRBench), so the "
                    + "manifest was not written. If you have reviewed the violations above and want "
                    + "the manifest anyway, re-run with -ignore_pairing_errors. Do NOT use "
                    + "-no_similarity_gate for this: it changes which sequences are generated and "
                    + "produces a library that should not be searched.");
        }
        Cloger.getInstance().logger.warn(
                "Entrapment manifest failed integrity checks: " + detail
                + ". Writing it anyway because -ignore_pairing_errors was given. A paired FDP "
                + "estimator may crash or report a mis-scaled FDP on this manifest.");
    }

    /** Render violations as a single message suitable for an exception or a log line. */
    public static String describe(List<Violation> violations) {
        StringBuilder sb = new StringBuilder();
        for (Violation v : violations) {
            if (sb.length() > 0) {
                sb.append("; ");
            }
            sb.append(v);
        }
        return sb.toString();
    }

    private static void add(List<String> list, String example) {
        if (list.size() < MAX_EXAMPLES) {
            list.add(example);
        }
    }

    /** A violation tally: full count, capped examples. */
    private static final class Counter {
        private int count;
        private final List<String> examples = new ArrayList<>();

        void hit(String example) {
            count++;
            add(examples, example);
        }

        void addTo(List<Violation> out, String kind) {
            if (count > 0) {
                out.add(new Violation(kind, count, examples));
            }
        }
    }
}
