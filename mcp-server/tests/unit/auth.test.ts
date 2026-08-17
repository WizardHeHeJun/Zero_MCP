// auth.ts 三态全分支测试：isLoopbackAddress / resolveEnforcedToken / verifyBearer。
//
// 纯函数、无 I/O，覆盖 mcp-integration.md 要求的鉴权边界纪律（loopback 判据、
// token 非 ASCII fail-fast、Bearer 头三态）。

import { describe, expect, it } from "vitest";
import { isLoopbackAddress, resolveEnforcedToken, verifyBearer } from "../../src/auth.js";

describe("isLoopbackAddress", () => {
  it.each([
    ["127.0.0.1", true],
    ["127.8.8.8", true], // 127.0.0.0/8 整段，不只 127.0.0.1
    ["127.255.255.255", true],
    ["::1", true],
    ["localhost", true],
    ["LOCALHOST", true], // 大小写不敏感
    ["  localhost  ", true], // 首尾空白容错
    ["0.0.0.0", false],
    ["::", false],
    ["192.168.1.10", false], // LAN IP
    ["10.0.0.5", false],
    ["8.8.8.8", false],
    ["256.0.0.1", false], // 非法八位组
    ["127.0.0", false], // 段数不对
    ["not-a-host", false], // 未知主机名保守视为非 loopback
    ["", false],
  ])("isLoopbackAddress(%s) -> %s", (host, expected) => {
    expect(isLoopbackAddress(host)).toBe(expected);
  });
});

describe("resolveEnforcedToken", () => {
  it("设了 token 时即便 host 是 loopback 也强制鉴权", () => {
    expect(resolveEnforcedToken("127.0.0.1", "secret-token")).toBe("secret-token");
  });

  it("token 前后空白被 trim", () => {
    expect(resolveEnforcedToken("127.0.0.1", "  secret-token  ")).toBe("secret-token");
  });

  it("未设 token 且 host 为 loopback → 免鉴权（返回 null）", () => {
    expect(resolveEnforcedToken("127.0.0.1", "")).toBeNull();
    expect(resolveEnforcedToken("::1", "")).toBeNull();
    expect(resolveEnforcedToken("localhost", "")).toBeNull();
  });

  it("未设 token 且 host 全为空白 → 视同未设，loopback 时仍免鉴权", () => {
    expect(resolveEnforcedToken("127.0.0.1", "   ")).toBeNull();
  });

  it("未设 token 且 host 非 loopback → fail-fast 抛错", () => {
    expect(() => resolveEnforcedToken("0.0.0.0", "")).toThrow(/非 loopback/);
    expect(() => resolveEnforcedToken("192.168.1.10", "")).toThrow(/未设/);
  });

  it("token 非纯 ASCII → fail-fast 抛错（即便 host 是 loopback）", () => {
    expect(() => resolveEnforcedToken("127.0.0.1", "秘密令牌")).toThrow(/纯 ASCII/);
  });

  it("token 为纯 ASCII（hex/UUID/base64 风格）→ 正常通过", () => {
    expect(resolveEnforcedToken("0.0.0.0", "a1b2c3d4-uuid-like_token+/=")).toBe(
      "a1b2c3d4-uuid-like_token+/=",
    );
  });
});

describe("verifyBearer", () => {
  it("enforcedToken 为 null 时恒放行，即便没有 Authorization 头", () => {
    expect(verifyBearer(undefined, null)).toBe(true);
    expect(verifyBearer("Bearer whatever", null)).toBe(true);
  });

  it("缺 Authorization 头 → 拒绝", () => {
    expect(verifyBearer(undefined, "secret")).toBe(false);
  });

  it("非 Bearer 格式的头 → 拒绝", () => {
    expect(verifyBearer("Basic dXNlcjpwYXNz", "secret")).toBe(false);
    expect(verifyBearer("secret", "secret")).toBe(false); // 缺 "Bearer " 前缀
  });

  it("Bearer 前缀大小写不敏感", () => {
    expect(verifyBearer("bearer secret", "secret")).toBe(true);
    expect(verifyBearer("BEARER secret", "secret")).toBe(true);
  });

  it("token 不匹配 → 拒绝", () => {
    expect(verifyBearer("Bearer wrong-token", "secret")).toBe(false);
  });

  it("token 长度不同也要能正确判否（常量时间比较分支覆盖）", () => {
    expect(verifyBearer("Bearer short", "much-longer-secret-token")).toBe(false);
  });

  it("token 匹配 → 通过", () => {
    expect(verifyBearer("Bearer secret", "secret")).toBe(true);
  });

  it("空字符串 token 部分（Bearer 后无内容）→ 视为空字符串比较", () => {
    expect(verifyBearer("Bearer ", "secret")).toBe(false);
  });
});
