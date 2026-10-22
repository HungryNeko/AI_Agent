import React, { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

// Small UI primitives shared by every page: popovers opened from pills, step sliders and
// resizable / collapsible panes.

function readStored(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writeStored(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage can be unavailable (private mode); the layout just is not remembered */
  }
}

/** A pill-shaped button that opens a floating panel (up by default, since pills live in the composer). */
export function PopPill({
  icon = null,
  label = null,
  title = "",
  tone = "",
  active = false,
  caret = false,
  align = "left",
  direction = "up",
  width = 320,
  className = "",
  children,
  onOpenChange,
  iconOnly = false,
}) {
  const [open, setOpenState] = useState(false);
  const rootRef = useRef(null);

  const setOpen = useCallback(
    (value) => {
      setOpenState((current) => {
        const next = typeof value === "function" ? value(current) : value;
        if (next !== current) onOpenChange?.(next);
        return next;
      });
    },
    [onOpenChange],
  );

  useEffect(() => {
    if (!open) return undefined;
    const onPointer = (event) => {
      if (rootRef.current && !rootRef.current.contains(event.target)) setOpen(false);
    };
    const onKey = (event) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, setOpen]);

  const body = typeof children === "function" ? children({ close: () => setOpen(false) }) : children;
  return (
    <div className={`popWrap ${className}`} ref={rootRef}>
      <button
        type="button"
        className={`pill${tone ? ` tone-${tone}` : ""}${open ? " open" : ""}${active ? " active" : ""}${iconOnly ? " iconOnly" : ""}`}
        title={title}
        aria-label={title || undefined}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        {icon}
        {label !== null && label !== "" && <span className="pillLabel">{label}</span>}
        {caret && <span className="pillCaret" aria-hidden="true">▾</span>}
      </button>
      {open && (
        <div
          className={`popPanel ${direction} align-${align}`}
          style={{ "--pop-width": `${width}px` }}
          role="dialog"
          aria-label={title}
        >
          {body}
        </div>
      )}
    </div>
  );
}

/** A plain pill button that does something (no panel). */
export function Pill({ icon = null, label = null, title = "", tone = "", active = false, onClick, disabled = false, className = "", type = "button" }) {
  return (
    <button
      type={type}
      className={`pill${tone ? ` tone-${tone}` : ""}${active ? " active" : ""}${className ? ` ${className}` : ""}`}
      title={title}
      aria-label={title || undefined}
      aria-pressed={active || undefined}
      onClick={onClick}
      disabled={disabled}
    >
      {icon}
      {label !== null && label !== "" && <span className="pillLabel">{label}</span>}
    </button>
  );
}

/** One row of a popover menu. */
export function MenuItem({ icon = null, label, hint = "", selected = false, onClick, disabled = false, danger = false }) {
  return (
    <button
      type="button"
      role="menuitem"
      className={`menuItem${selected ? " selected" : ""}${danger ? " danger" : ""}`}
      onClick={onClick}
      disabled={disabled}
    >
      {icon && <span className="menuIcon">{icon}</span>}
      <span className="menuText">
        <span className="menuLabel">{label}</span>
        {hint && <small>{hint}</small>}
      </span>
      {selected && <span className="menuCheck" aria-hidden="true">✓</span>}
    </button>
  );
}

/** A slider with a fixed number of named steps. steps: [{ value, label, hint? }]. */
export function StepSlider({ title, steps, value, onChange, tone = "", ariaLabel = "" }) {
  const found = steps.findIndex((step) => step.value === value);
  const index = found < 0 ? 0 : found;
  const last = Math.max(1, steps.length - 1);
  return (
    <div className={`stepSlider${tone ? ` tone-${tone}` : ""}`}>
      <div className="stepHead">
        <span>{title}</span>
        <strong>{steps[index]?.label}</strong>
      </div>
      <input
        className="stepRange"
        type="range"
        min="0"
        max={steps.length - 1}
        step="1"
        value={index}
        onChange={(event) => onChange(steps[Number(event.target.value)].value)}
        aria-label={ariaLabel || title}
        aria-valuetext={steps[index]?.label}
        style={{ "--fill": `${(index / last) * 100}%` }}
      />
      <div className="stepTicks" style={{ gridTemplateColumns: `repeat(${steps.length}, 1fr)` }}>
        {steps.map((step, position) => (
          <button key={step.value} type="button" className={position === index ? "active" : ""} onClick={() => onChange(step.value)}>
            {step.label}
          </button>
        ))}
      </div>
      {steps[index]?.hint && <p className="stepHint">{steps[index].hint}</p>}
    </div>
  );
}

/** Segmented control: small mutually exclusive buttons in one row. */
export function Segmented({ options, value, onChange, ariaLabel = "" }) {
  return (
    <div className="segmented" role="group" aria-label={ariaLabel}>
      {options.map((option) => (
        <button key={option.value} type="button" className={option.value === value ? "active" : ""} onClick={() => onChange(option.value)} title={option.title || undefined}>
          {option.label}
        </button>
      ))}
    </div>
  );
}

/**
 * Width of one resizable pane, remembered across reloads.
 * side "start": the pane is before the divider (drag right = wider); "end": after it.
 */
export function usePane(key, { size = 300, min = 200, max = 640, side = "start", collapsed = false } = {}) {
  const [state, setState] = useState(() => {
    const saved = readStored(`pane:${key}`, null);
    // On a phone-sized window side panels start hidden; the divider tab brings them back.
    const narrow = typeof window !== "undefined" && window.innerWidth < 760;
    return {
      size: Math.min(max, Math.max(min, Number(saved?.size) || size)),
      collapsed: typeof saved?.collapsed === "boolean" ? saved.collapsed : collapsed || narrow,
    };
  });
  useEffect(() => {
    writeStored(`pane:${key}`, state);
  }, [key, state]);
  const setSize = useCallback((next) => setState((current) => ({ ...current, size: Math.min(max, Math.max(min, Math.round(next))) })), [min, max]);
  const toggle = useCallback(() => setState((current) => ({ ...current, collapsed: !current.collapsed })), []);
  const setCollapsed = useCallback((value) => setState((current) => ({ ...current, collapsed: Boolean(value) })), []);
  return { ...state, min, max, side, setSize, toggle, setCollapsed, width: state.collapsed ? 0 : state.size };
}

/** The draggable divider between two panes. Double-click or the little tab hides / shows the pane. */
export function Resizer({ pane, label = "" }) {
  const dragRef = useRef(null);

  function onPointerDown(event) {
    if (event.button !== 0 || event.target.closest(".resizerTab")) return;
    event.preventDefault();
    const start = { x: event.clientX, size: pane.collapsed ? pane.min : pane.size };
    dragRef.current = start;
    if (pane.collapsed) pane.setCollapsed(false);
    event.currentTarget.setPointerCapture?.(event.pointerId);
    document.body.classList.add("isResizing");
  }

  function onPointerMove(event) {
    const start = dragRef.current;
    if (!start) return;
    const delta = event.clientX - start.x;
    pane.setSize(pane.side === "start" ? start.size + delta : start.size - delta);
  }

  function onPointerUp(event) {
    if (!dragRef.current) return;
    dragRef.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    document.body.classList.remove("isResizing");
  }

  function onKeyDown(event) {
    const step = event.shiftKey ? 40 : 12;
    if (event.key === "ArrowLeft") pane.setSize(pane.size + (pane.side === "start" ? -step : step));
    else if (event.key === "ArrowRight") pane.setSize(pane.size + (pane.side === "start" ? step : -step));
    else if (event.key === "Enter") pane.toggle();
    else return;
    event.preventDefault();
  }

  // The arrow points the way the pane will move when clicked.
  const pointsLeft = pane.side === "start" ? !pane.collapsed : pane.collapsed;
  return (
    <div
      className={`resizer${pane.collapsed ? " collapsed" : ""}`}
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={pane.toggle}
      onKeyDown={onKeyDown}
    >
      <button
        type="button"
        className="resizerTab"
        onClick={pane.toggle}
        title={pane.collapsed ? "Show" : "Hide"}
        aria-label={pane.collapsed ? "Show panel" : "Hide panel"}
      >
        {pointsLeft ? <ChevronLeft size={13} /> : <ChevronRight size={13} />}
      </button>
    </div>
  );
}

/** Grid template for a row of panes and dividers; `null` marks the flexible pane. */
export function splitColumns(...parts) {
  return parts
    .map((part) => {
      if (part === null) return "minmax(0, 1fr)";
      if (part === "r") return "var(--resizer)";
      return `${part.width}px`;
    })
    .join(" ");
}
