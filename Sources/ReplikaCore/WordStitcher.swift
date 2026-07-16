import Foundation

/// Concatenates per-window aligned words (already in absolute time) into one
/// monotonic list, de-duplicating words that fall inside the overlap region
/// shared with the previous window at a force-split boundary.
public enum WordStitcher {
    public static func stitch(perWindow: [(window: ChunkWindow, words: [Word])]) -> [Word] {
        var result: [Word] = []
        var previousEnd: Double?
        for (window, words) in perWindow {
            if window.overlapsPrevious, let boundary = previousEnd {
                for word in words where (word.start + word.end) / 2 >= boundary {
                    result.append(word)
                }
            } else {
                result.append(contentsOf: words)
            }
            previousEnd = window.end
        }
        return result
    }
}
