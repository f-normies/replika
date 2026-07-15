import Testing
import Foundation
@testable import ReplikaCore

private struct StubProvider: TranscriptionProvider {
    let capabilities = ProviderCaps(diarization: false, wordTimestamps: false, streaming: false)
    func transcribe(_ audio: AudioSource,
                    options: TranscribeOptions) -> AsyncThrowingStream<TranscriptEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func cancel() {}
}

@Test func registersAndMakesProvider() {
    var reg = ProviderRegistry()
    reg.register("stub") { StubProvider() }
    #expect(reg.make("stub") != nil)
    #expect(reg.make("missing") == nil)
    #expect(reg.names == ["stub"])
}
