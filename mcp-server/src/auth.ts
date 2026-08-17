// Zero_MCP · TS 聚合器 HTTP 传输的 Bearer 鉴权（纯函数，边界层）。
//
// 三态语义与 Zero `src/mcp_server/auth.py::resolve_enforced_token` 对齐（见本仓
// README.md「zero-link · Zero MCP Client」一节的鉴权三态说明，及本文件对应新增
// 小节「TS MCP 对外聚合层」）：
//   ① 设了 token → 恒强制鉴权（即便 loopback），token 须纯 ASCII，否则 fail-fast
//     （非 ASCII 密钥经 HTTP 头编码歧义会恒 401，宁可启动期拒绝也不留一个永远
//     401 的死配置）。
//   ② 未设 token 且 host 为 loopback → 免鉴权（本机零回归）。loopback 判定含
//     127.0.0.0/8、::1、"localhost"——"localhost" 归入 loopback 是相对 Zero 侧
//     Python 先例的显式扩展（Zero 侧同判据同名集合，此处对齐，不是新发明）。
//   ③ 未设 token 且非 loopback → fail-fast，拒绝对外开无鉴权裸端口。
//
// 本文件是「传输层不塞业务逻辑」的落点之一：只做鉴权判定，不出现任何
// VTS/桌面业务语义。

import { timingSafeEqual } from "node:crypto";

const LOOPBACK_HOSTNAMES = new Set(["localhost"]);

/** 校验字符串是否为合法的 IPv4 十进制八位组（0-255，无前导零歧义不做强校验）。 */
function isValidIPv4Octet(octet: string): boolean {
  if (octet.length === 0 || octet.length > 3) {
    return false;
  }
  if (!/^\d+$/.test(octet)) {
    return false;
  }
  const value = Number(octet);
  return value >= 0 && value <= 255;
}

/** host 是否落在 127.0.0.0/8（IPv4 loopback 段）。 */
function isIPv4Loopback(host: string): boolean {
  const parts = host.split(".");
  if (parts.length !== 4) {
    return false;
  }
  if (!parts.every(isValidIPv4Octet)) {
    return false;
  }
  return Number(parts[0]) === 127;
}

/**
 * 判定 host 是否 loopback（127.0.0.0/8、::1、"localhost"）。
 *
 * 未知主机名一律 False（保守要鉴权），与 Zero `_is_loopback` 同判据。
 */
export function isLoopbackAddress(host: string): boolean {
  const normalized = host.trim().toLowerCase();
  if (LOOPBACK_HOSTNAMES.has(normalized)) {
    return true;
  }
  if (normalized === "::1") {
    return true;
  }
  return isIPv4Loopback(normalized);
}

/**
 * 决定是否强制鉴权 + 用哪个 token（可测纯函数，镜像 Zero `resolve_enforced_token`）。
 *
 * @param host  HTTP 监听 host（ZERO_MCP_AGGREGATOR_HOST）。
 * @param token 配置的 token（未设时传空字符串）。
 * @returns 强制鉴权时返回 trim 后的 token；免鉴权返回 null。
 * @throws {Error} token 非纯 ASCII，或未设 token 且 host 非 loopback。
 */
export function resolveEnforcedToken(host: string, token: string): string | null {
  const stripped = token.trim();
  if (stripped.length > 0) {
    if (!/^[\x00-\x7F]*$/.test(stripped)) {
      throw new Error(
        "ZERO_MCP_AGGREGATOR_TOKEN 须为纯 ASCII（hex/UUID/base64 等）；" +
          "非 ASCII 密钥经 HTTP 头编码歧义会恒 401",
      );
    }
    return stripped;
  }
  if (isLoopbackAddress(host)) {
    return null;
  }
  throw new Error(
    `ZERO_MCP_AGGREGATOR_HOST=${host} 非 loopback 却未设 ZERO_MCP_AGGREGATOR_TOKEN——` +
      "拒绝对外开无鉴权裸端口；请设 token 或绑回 127.0.0.1",
  );
}

/** 从 `Authorization` 头取出 Bearer token；无/非 Bearer 头 → undefined。 */
function extractBearerToken(header: string | undefined): string | undefined {
  if (header === undefined) {
    return undefined;
  }
  const prefix = "bearer ";
  if (header.slice(0, prefix.length).toLowerCase() !== prefix) {
    return undefined;
  }
  return header.slice(prefix.length);
}

/** 常量时间比较两个字符串是否相等（避免计时旁路，先比长度再比内容）。 */
function timingSafeEqualString(a: string, b: string): boolean {
  const bufA = Buffer.from(a, "utf-8");
  const bufB = Buffer.from(b, "utf-8");
  if (bufA.length !== bufB.length) {
    return false;
  }
  return timingSafeEqual(bufA, bufB);
}

/**
 * 校验请求的 `Authorization` 头是否满足鉴权要求。
 *
 * @param header         请求的 `Authorization` 头原始值（可能 undefined）。
 * @param enforcedToken  `resolveEnforcedToken` 的结果；null = 免鉴权，恒通过。
 * @returns 是否通过鉴权。
 */
export function verifyBearer(header: string | undefined, enforcedToken: string | null): boolean {
  if (enforcedToken === null) {
    return true;
  }
  const provided = extractBearerToken(header);
  if (provided === undefined) {
    return false;
  }
  return timingSafeEqualString(provided, enforcedToken);
}
