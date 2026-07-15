public struct ProviderRegistry {
    public typealias Factory = @Sendable () -> TranscriptionProvider

    private var factories: [String: Factory] = [:]

    public init() {}

    public mutating func register(_ name: String, _ factory: @escaping Factory) {
        factories[name] = factory
    }

    public func make(_ name: String) -> TranscriptionProvider? {
        factories[name]?()
    }

    public var names: [String] { factories.keys.sorted() }
}
