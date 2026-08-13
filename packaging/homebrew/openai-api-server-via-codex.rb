class OpenaiApiServerViaCodex < Formula
  desc "OpenAI-compatible local API server backed by Codex credentials"
  homepage "https://github.com/hotchpotch/openai-api-server-via-codex"
  url "https://github.com/hotchpotch/openai-api-server-via-codex/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "e9d0f078b4430631dd18e60cb37700cd64bb61c9c983d20300550125bb019538"
  license "Apache-2.0"

  livecheck do
    url :stable
    regex(/^v?(\d+(?:\.\d+)+)$/i)
  end

  depends_on "go" => :build

  def install
    ldflags = "-s -w -X main.version=#{version}"
    system "go", "build", *std_go_args(ldflags:), "./cmd/openai-api-server-via-codex"
  end

  test do
    assert_equal version.to_s, shell_output("#{bin}/openai-api-server-via-codex --version").strip
    config = shell_output("#{bin}/openai-api-server-via-codex config-generate --stdout")
    assert_match "port = 18080", config
  end
end
