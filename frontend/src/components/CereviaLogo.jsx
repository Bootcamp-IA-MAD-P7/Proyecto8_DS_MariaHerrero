function CereviaLogo({ compact = false }) {
  return (
    <span
      className={`cerevia-logo${compact ? " cerevia-logo--compact" : ""}`}
      aria-label="CEREVIA"
    >
      <svg
        className="cerevia-logo__mark"
        viewBox="0 0 64 64"
        aria-hidden="true"
      >
        <path d="M31.8 8.5c-7.4-4.3-16.7.6-16.7 9.4 0 1 .1 2 .4 2.9-5.7 2-8.5 8.5-5.7 13.7-3.2 5.7.2 12.9 6.5 14.3 1.7 7 10.4 9.3 15.5 4.2V8.5Z" />
        <path d="M32.2 8.5c7.4-4.3 16.7.6 16.7 9.4 0 1-.1 2-.4 2.9 5.7 2 8.5 8.5 5.7 13.7 3.2 5.7-.2 12.9-6.5 14.3-1.7 7-10.4 9.3-15.5 4.2V8.5Z" />
        <path d="M22 18.5h5.8m-8.7 9.2h8.7m-11 9.1h11m-5.8 9h5.8M42 18.5h-5.8m8.7 9.2h-8.7m11 9.1h-11m5.8 9h-5.8" />
        <circle cx="19" cy="18.5" r="2" />
        <circle cx="16" cy="27.7" r="2" />
        <circle cx="13.8" cy="36.8" r="2" />
        <circle cx="19" cy="45.8" r="2" />
        <circle cx="45" cy="18.5" r="2" />
        <circle cx="48" cy="27.7" r="2" />
        <circle cx="50.2" cy="36.8" r="2" />
        <circle cx="45" cy="45.8" r="2" />
      </svg>

      <span className="cerevia-logo__copy">
        <strong>CEREVIA</strong>
        {!compact && (
          <span>Inteligencia que ayuda a anticiparse.</span>
        )}
      </span>
    </span>
  )
}

export default CereviaLogo
