import Testing
import Foundation
import ReplikaCore
@testable import SpeechSwiftProvider

// PRIVACY: this suite intentionally does NOT bundle any audio resource.
// The clip path comes from the `SMOKE_AUDIO` env var, and the test only
// runs when BOTH `RUN_MODEL_TESTS == "1"` AND `SMOKE_AUDIO` are set to a
// non-empty value — otherwise it is skipped, keeping the offline suite green.
private func smokeAudioURL() -> URL? {
    let env = ProcessInfo.processInfo.environment
    guard env["RUN_MODEL_TESTS"] == "1",
          let path = env["SMOKE_AUDIO"], !path.isEmpty else {
        return nil
    }
    return URL(fileURLWithPath: path)
}

@Test(.enabled(if: smokeAudioURL() != nil))
func smokeProducesLabeledSegments() async throws {
    let url = try #require(smokeAudioURL())
    let provider = SpeechSwiftProvider()

    var done: Transcript?
    var sawSpeaker = false
    var speakerIDs = Set<Int>()
    for try await ev in provider.transcribe(.file(url), options: TranscribeOptions(diarize: true)) {
        switch ev {
        case .speaker(let tag):
            sawSpeaker = true
            speakerIDs.insert(tag.speaker)
        case .done(let t): done = t
        default: break
        }
    }

    // Distinct-speaker count is a metric, not transcript content — safe to
    // surface for the spike's key result (diarization signal on the RU clip).
    print("SMOKE: distinct speakers = \(speakerIDs.count)")

    let transcript = try #require(done)
    #expect(!transcript.segments.isEmpty)
    #expect(transcript.segments.allSatisfy { !$0.text.isEmpty })
    #expect(sawSpeaker)
    // Guards against a silent `SpeakerMerger` failure: raw `.speaker` stream
    // events (`sawSpeaker` above) only prove diarization ran, not that the
    // merge step actually attached labels to the final segments.
    #expect(transcript.segments.contains(where: { $0.speaker != nil }))
    for seg in transcript.segments where seg.words.count > 1 {
        for i in 1..<seg.words.count {
            #expect(seg.words[i].start >= seg.words[i - 1].start)  // monotonic
            #expect(seg.words[i].end <= seg.end + 0.001)           // within segment
        }
    }
}

@Test(.enabled(if: smokeAudioURL() != nil))
func longFormProducesManySegments() async throws {
    let url = try #require(smokeAudioURL())
    let samples = try AudioLoader.loadMono16k(url)
    let durationSec = Double(samples.count) / 16000.0

    let provider = SpeechSwiftProvider()
    var done: Transcript?
    for try await ev in provider.transcribe(.file(url), options: TranscribeOptions(diarize: true)) {
        if case .done(let t) = ev { done = t }
    }
    let transcript = try #require(done)

    // Regression on FINDINGS §5: single-shot transcribe() returned 0 segments
    // on the ~9.4 min clip. Chunking must yield many labeled segments.
    #expect(transcript.segments.count >= 5)

    // Cross-segment monotonicity and coverage reaching well into the clip.
    let starts = transcript.segments.map(\.start)
    #expect(starts == starts.sorted())
    if let last = transcript.segments.last {
        #expect(last.end > durationSec * 0.5)
    }

    // Metrics only — never transcript text.
    let coverage = (transcript.segments.last?.end ?? 0) / durationSec * 100
    print("LONGFORM: segments=\(transcript.segments.count), coverage=\(String(format: "%.0f%%", coverage))")
}
