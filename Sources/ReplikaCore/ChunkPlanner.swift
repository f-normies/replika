import Foundation

/// A bounded transcription window in absolute seconds. `overlapsPrevious` is
/// true only for a force-split sub-window whose start was pulled back to
/// overlap the previous sub-window of the SAME speech span.
public struct ChunkWindow: Sendable, Equatable {
    public let start: Double
    public let end: Double
    public let overlapsPrevious: Bool
    public init(start: Double, end: Double, overlapsPrevious: Bool) {
        self.start = start; self.end = end; self.overlapsPrevious = overlapsPrevious
    }
}

/// Turns VAD speech spans into transcription windows, force-splitting any span
/// longer than `config.maxChunkSeconds` into overlapping sub-windows.
public enum ChunkPlanner {
    public static func plan(spans: [(start: Double, end: Double)],
                            config: ChunkConfig) -> [ChunkWindow] {
        // A non-positive max would leave every long span un-splittable; emit
        // each span as a single window so the loop below can never spin
        // without advancing.
        guard config.maxChunkSeconds > 0 else {
            return spans.map { ChunkWindow(start: $0.start, end: $0.end, overlapsPrevious: false) }
        }
        // Clamp the overlap so each sub-window strictly advances: a configured
        // overlap >= maxChunkSeconds (degenerate tuning) would otherwise leave
        // winStart stationary and loop forever. Valid overlaps pass through.
        let effectiveOverlap = config.overlapSeconds < config.maxChunkSeconds
            ? max(config.overlapSeconds, 0)
            : config.maxChunkSeconds / 2

        var windows: [ChunkWindow] = []
        for span in spans {
            if span.end - span.start <= config.maxChunkSeconds {
                windows.append(ChunkWindow(start: span.start, end: span.end,
                                           overlapsPrevious: false))
                continue
            }
            var winStart = span.start
            while winStart < span.end {
                let winEnd = min(winStart + config.maxChunkSeconds, span.end)
                windows.append(ChunkWindow(start: winStart, end: winEnd,
                                           overlapsPrevious: winStart > span.start))
                if winEnd >= span.end { break }
                winStart = winEnd - effectiveOverlap
            }
        }
        return windows
    }
}
