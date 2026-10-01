/**
 * "Under development" — the console's mark for a field no model produces yet.
 *
 * Used wherever the K.I.R.A. layout has a slot our pipeline cannot fill
 * honestly: branch hops and volume, attention saliency, kill-chain lanes no
 * head emits. The slot stays, so the layout does not shift when a model
 * starts producing it; the value is never guessed. Each use names what is
 * missing in `title`, and docs/DASHBOARD_INTEGRATION.md lists them all.
 */

import type { CSSProperties, ReactNode } from "react";

export default function UnderDev({
  title,
  children = "Under development",
  block,
  style,
}: {
  /** Tooltip: what is missing and why. */
  title?: string;
  children?: ReactNode;
  /** Fill the container as an empty-state panel instead of an inline mark. */
  block?: boolean;
  style?: CSSProperties;
}) {
  return (
    <span className={block ? "udev is-block" : "udev"} title={title} style={style}>
      {children}
    </span>
  );
}
