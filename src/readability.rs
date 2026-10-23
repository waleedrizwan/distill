//! How hard a sentence is to read, for `--plain`.
//!
//! The Flesch-Kincaid grade of the sentence alone: 0.39 times the words per
//! sentence, plus 11.8 times the syllables per word, minus 15.59. Syllables
//! are vowel groups (a final silent "e" dropped), as `eval/score.py` counts. Long
//! sentences and long words both raise it. It is a measure of form, not of
//! meaning: it cannot tell a familiar long word from an obscure short one.

/// Syllables in a word, by vowel groups.
pub fn syllables(word: &str) -> usize {
    let w = word.to_lowercase();
    let mut groups = 0;
    let mut in_vowel = false;
    for c in w.chars() {
        let v = matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y');
        if v && !in_vowel {
            groups += 1;
        }
        in_vowel = v;
    }
    if w.ends_with('e') && !w.ends_with("le") && !w.ends_with("ee") && !w.ends_with("ye") && groups > 1 {
        groups -= 1;
    }
    groups.max(1)
}

/// Difficulty for ranking: the sentence's grade, except that text which does
/// not end like a sentence (a PDF fragment such as "... was prepared following
/// the") counts as hardest. Short fragments would otherwise look easiest.
pub fn difficulty(sentence: &str) -> f32 {
    let t = sentence.trim_end().trim_end_matches(['"', '\'', ')', ']', '\u{201D}', '\u{2019}']);
    if t.ends_with(['.', '!', '?']) {
        grade(sentence)
    } else {
        f32::MAX / 4.0
    }
}

/// The Flesch-Kincaid grade of one sentence (0 for a sentence with no words).
pub fn grade(sentence: &str) -> f32 {
    let words: Vec<&str> = sentence
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return 0.0;
    }
    let syl: usize = words.iter().map(|w| syllables(w)).sum();
    0.39 * words.len() as f32 + 11.8 * syl as f32 / words.len() as f32 - 15.59
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syllable_counts() {
        for (w, n) in [("cat", 1), ("table", 2), ("make", 1), ("reading", 2), ("simplification", 5), ("free", 1), ("rhythm", 1)] {
            assert_eq!(syllables(w), n, "{w}");
        }
    }

    #[test]
    fn plain_sentences_grade_lower() {
        let plain = "The cat sat on the mat and looked at the dog.";
        let hard = "Heterogeneous microstructural characteristics substantially influence magnetization reversal.";
        assert!(grade(plain) < grade(hard));
        assert_eq!(grade(""), 0.0);
        assert_eq!(grade("123 456"), 0.0);
    }

    #[test]
    fn fragments_are_hardest() {
        assert!(difficulty("It was prepared following the") > difficulty("Heterogeneous microstructural characteristics substantially influence magnetization reversal."));
        assert_eq!(difficulty("The cat sat."), grade("The cat sat."));
        assert_eq!(difficulty("\"The cat sat.\""), grade("\"The cat sat.\""));
    }

    #[test]
    fn matches_the_scorer() {
        // eval/score.py on the same sentence: 9 words, 11 vowel groups ("baked" counts two).
        let g = grade("We baked bread with flour, water, salt and yeast.");
        assert!((g - (0.39 * 9.0 + 11.8 * 11.0 / 9.0 - 15.59)).abs() < 1e-4, "{g}");
    }
}
