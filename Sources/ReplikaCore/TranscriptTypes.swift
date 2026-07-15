import Foundation

public struct Word: Sendable, Equatable {
    public let text: String
    public let start: Double
    public let end: Double
    public init(text: String, start: Double, end: Double) {
        self.text = text; self.start = start; self.end = end
    }
}

public struct Segment: Sendable, Equatable {
    public let text: String
    public let start: Double
    public let end: Double
    public let words: [Word]
    public var speaker: Int?
    public init(text: String, start: Double, end: Double, words: [Word], speaker: Int? = nil) {
        self.text = text; self.start = start; self.end = end
        self.words = words; self.speaker = speaker
    }
}

public struct SpeakerTag: Sendable, Equatable {
    public let speaker: Int
    public let start: Double
    public let end: Double
    public init(speaker: Int, start: Double, end: Double) {
        self.speaker = speaker; self.start = start; self.end = end
    }
}

public struct Transcript: Sendable, Equatable {
    public let segments: [Segment]
    public let language: String?
    public init(segments: [Segment], language: String? = nil) {
        self.segments = segments; self.language = language
    }
}

public enum TranscriptEvent: Sendable {
    case progress(Double)
    case committed(Segment)
    case speaker(SpeakerTag)
    case done(Transcript)
}
