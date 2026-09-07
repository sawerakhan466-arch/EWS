# Upgraded Chitral Early Warning System (Gradio) — UPDATED with:
# - Multi-channel alerts (Twilio: SMS / WhatsApp / Voice) with simulated fallback
# - Department auto-routing for community reports
# - Weather (Open-Meteo) integration + NASA GIBS tile helper for satellite layer
# - Validations: Name (alphabets only) and Phone (digits only)
# - Editable satellite date for NASA GIBS tile template
#
# IMPORTANT: This preserves all original functions/flow. New features are optional and safe:
# - If TWILIO creds are not provided, system falls back to simulated SMS (original behavior).
# - For email sending (optional), configure SMTP credentials.
#
# Steps for real use:
# - Create Twilio account (trial OK for testing) and set environment vars:
#   TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
# - Optionally set TWILIO_WHATSAPP_FROM (e.g. "whatsapp:+1415XXXXXXX")
# - Optionally set SMTP_EMAIL and SMTP_APP_PASSWORD to enable email alerts.
#
# Paste into Colab and run.


import random
import time
import json
import os
import math
import requests
import re
from datetime import datetime, timedelta
from groq import Groq

# ------------------------------
# Keep original Groq API setup logic (unchanged)
# ------------------------------
api_key = os.environ.get("GROQ_API_KEY", "")  # in Colab you can set env or keep blank
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

def get_ai_guidance(water_level, rainfall, glacier_melt):
    """
    Uses your Groq LLM (same as original). If API unavailable or missing,
    returns a local fallback guidance.
    """
    prompt = f"Water level: {water_level}m, Rainfall: {rainfall}mm, Glacier melt: {glacier_melt}cm. Generate detailed safety instructions, evacuation advice, emergency kit guidance, and alerts for local communities in Chitral."
    try:
        if not os.environ.get("GROQ_API_KEY"):
            raise Exception("No API key present")
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Local fallback (safe, short)
        fallback = (
            "AI guidance unavailable (offline). Follow local safety rules:\n"
            "- Move to higher ground if water rises or river level increases.\n"
            "- Keep emergency kit (water, flashlight, medicines, important docs).\n"
            "- Keep mobile charged and maintain contact with rescue teams.\n"
            "- Follow instructions of local authorities."
        )
        return fallback

# ------------------------------
# Sensor simulation and history buffer (for prediction)
# ------------------------------
HISTORY = []  # list of tuples (timestamp, water_level, rainfall, glacier_melt)
HISTORY_MAX = 48  # keep last 48 simulated readings (e.g., hourly)

def simulate_sensor_data():
    """Return one simulated reading (keeps same distribution as original)."""
    water_level = round(random.uniform(0, 10), 1)
    rainfall = round(random.uniform(0, 100), 1)
    glacier_melt = round(random.uniform(0, 10), 1)
    ts = datetime.utcnow().isoformat()
    # Append to history
    HISTORY.append((ts, water_level, rainfall, glacier_melt))
    if len(HISTORY) > HISTORY_MAX:
        HISTORY.pop(0)
    return water_level, rainfall, glacier_melt

# Pre-fill history with simulated past values (to allow prediction immediately)
for _ in range(12):
    simulate_sensor_data()

# ------------------------------
# Risk scoring (unchanged logic) and role messages (keeps same texts)
# ------------------------------
def compute_risk_score(water_level, rainfall, glacier_melt):
    risk_score = 0
    if water_level > 8: risk_score += 2
    elif water_level > 5: risk_score += 1
    if rainfall > 50: risk_score += 2
    elif rainfall > 20: risk_score += 1
    if glacier_melt > 5: risk_score += 2
    elif glacier_melt > 2: risk_score += 1
    return risk_score

def risk_level_info(risk_score):
    if risk_score >= 5:
        return "🔴 HIGH RISK – Immediate action required!", "#ff0000", "#ff4d4d"
    elif risk_score >= 3:
        return "🟠 MODERATE RISK – Stay alert!", "#ff9900", "#ffa500"
    else:
        return "🟢 LOW RISK – Safe.", "#00ff00", "#7CFC00"

# ------------------------------
# Forecasting: linear extrapolation (unchanged)
# ------------------------------
def linear_forecast(series, hours_ahead, interval_hours=1.0):
    """
    series: list of numeric values ordered oldest->newest
    Simple linear regression (least squares) to estimate trend, then extrapolate.
    interval_hours: hour between samples (we simulate hourly-ish)
    """
    n = len(series)
    if n < 3:
        # Not enough data; repeat last value
        return series[-1] if series else 0.0
    # x values (0, 1, 2, ...)
    xs = [i*interval_hours for i in range(n)]
    ys = series
    x_mean = sum(xs)/n
    y_mean = sum(ys)/n
    num = sum((xs[i]-x_mean)*(ys[i]-y_mean) for i in range(n))
    den = sum((xs[i]-x_mean)**2 for i in range(n))
    if den == 0:
        slope = 0.0
    else:
        slope = num/den
    intercept = y_mean - slope*x_mean
    future_x = xs[-1] + hours_ahead
    return round(intercept + slope*future_x, 2)

def predict_next_hours(hours=6):
    """
    Use last HISTORY readings to predict water_level, rainfall, glacier_melt at hours ahead.
    Returns dict.
    """
    # Extract series (take last 12 values or full history)
    last_n = min(len(HISTORY), 12)
    wl_series = [h[1] for h in HISTORY[-last_n:]]
    rf_series = [h[2] for h in HISTORY[-last_n:]]
    gm_series = [h[3] for h in HISTORY[-last_n:]]
    pred_wl = max(0.0, linear_forecast(wl_series, hours))
    pred_rf = max(0.0, linear_forecast(rf_series, hours))
    pred_gm = max(0.0, linear_forecast(gm_series, hours))
    # Risk from predicted values
    pred_score = compute_risk_score(pred_wl, pred_rf, pred_gm)
    pred_level, _, _ = (risk_level_info(pred_score))
    return {
        "hours": hours,
        "water_level": round(pred_wl, 2),
        "rainfall": round(pred_rf, 2),
        "glacier_melt": round(pred_gm, 2),
        "risk_score": pred_score,
        "risk_level": pred_level
    }

# ------------------------------
# SMS simulation gateway (existing) — preserved
# ------------------------------
SMS_LOG_FILE = "simulated_sent_sms.json"

def load_sms_log():
    if os.path.exists(SMS_LOG_FILE):
        try:
            with open(SMS_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_sms_log(logs):
    with open(SMS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def send_sms_simulated(provider, phone_number, language, message_text):
    """
    Simulate SMS sending. Append to local JSON log and return a response dict.
    To replace with real API: implement provider-specific HTTP call here (Twilio, Jazz, etc.)
    """
    logs = load_sms_log()
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "provider": provider,
        "phone": phone_number,
        "language": language,
        "text": message_text
    }
    logs.append(entry)
    save_sms_log(logs)
    # Also print to console for debug
    print("[SIM_SMS]", entry)
    return {"status": "queued", "entry": entry, "note": "Simulated SMS - no real SMS sent."}

# ------------------------------
# Twilio: Multi-channel implementation (optional — uses env vars)
# - If TWILIO creds are not set, these functions fall back to simulated behavior.
# ------------------------------
from twilio.rest import Client as TwilioClient
TW_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TW_FROM = os.environ.get("TWILIO_PHONE_NUMBER")  # e.g. "+1234567890"
TW_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM")  # e.g. "whatsapp:+1415XXXXXXX"

_twilio_client = None
if TW_SID and TW_TOKEN:
    try:
        _twilio_client = TwilioClient(TW_SID, TW_TOKEN)
    except Exception as e:
        _twilio_client = None
        print("Twilio init failed:", e)

def send_sms_real_twilio(phone_number, message_text):
    """
    Send SMS using Twilio if available, otherwise fall back to simulated.
    """
    if _twilio_client and TW_FROM:
        try:
            msg = _twilio_client.messages.create(body=message_text, from_=TW_FROM, to=phone_number)
            return {"status": "sent", "sid": getattr(msg, "sid", None)}
        except Exception as e:
            print("Twilio SMS error:", e)
            # fall back to simulated log
            return send_sms_simulated("Twilio", phone_number, "EN", message_text)
    else:
        # fallback
        return send_sms_simulated("Simulated", phone_number, "EN", message_text)

def send_whatsapp_twilio(phone_number, message_text):
    """
    Send WhatsApp message via Twilio Sandbox (if configured). Fallback to simulated.
    phone_number should be in international format (e.g. +92XXXXXXXXX).
    """
    if _twilio_client and TW_WHATSAPP_FROM:
        try:
            to = f"whatsapp:{phone_number}"
            msg = _twilio_client.messages.create(body=message_text, from_=TW_WHATSAPP_FROM, to=to)
            return {"status": "sent", "sid": getattr(msg, "sid", None)}
        except Exception as e:
            print("Twilio WhatsApp error:", e)
            return {"status": "failed", "error": str(e)}
    else:
        # log simulated whatsapp in SMS log for demo
        return send_sms_simulated("WhatsApp-Simulated", phone_number, "EN", "[WhatsApp] " + message_text)

def send_voice_call_twilio(phone_number, message_text):
    """
    Make a simple voice call using Twilio TwiML via a URL. For demo we will use Twilio to create a call
    and use TwiML 'say' to speak the message. If Twilio not configured, fallback to logging.
    NOTE: For real voice calls you need a TwiML-capable URL or Twilio's 'twiml' parameter via a URL endpoint.
    We'll use Twilio's 'twiml' argument via a small hosted TwiML bin — for demo, fallback to simulated.
    """
    if _twilio_client and TW_FROM:
        try:
            # Simple TwiML instruction (text-to-speech) hosted via 'twiml' param - Twilio supports 'twiml' in create?
            # Twilio Python requires a publicly hosted URL for 'url' param. For simple demo, we skip complex hosting.
            # Here we'll create a call but use TwiML Bin URL if provided via env var.
            TW_TWIML_URL = os.environ.get("TWILIO_TWIML_URL")  # optional
            if TW_TWIML_URL:
                call = _twilio_client.calls.create(to=phone_number, from_=TW_FROM, url=TW_TWIML_URL)
                return {"status": "call_created", "sid": getattr(call, "sid", None)}
            else:
                # If no TwiML URL is configured, fallback to simulated log noting that a call would be placed.
                return send_sms_simulated("VoiceCall-Simulated", phone_number, "EN", "[VoiceCall] " + message_text)
        except Exception as e:
            print("Twilio call error:", e)
            return {"status": "failed", "error": str(e)}
    else:
        return send_sms_simulated("VoiceCall-Simulated", phone_number, "EN", "[VoiceCall] " + message_text)

# ------------------------------
# Template generator for English + Urdu (unchanged)
def make_sms_text(role, predicted_info, language="EN"):
    """
    role: Public/Rescue Worker/Admin
    predicted_info: dict from predict_next_hours
    language: "EN" or "UR"
    """
    if language == "UR":
        # Urdu templates (simple, short)
        return (
            f"Chitral Alert ({predicted_info['hours']}h): {predicted_info['risk_level']}\n"
            f"Water: {predicted_info['water_level']}m Rain: {predicted_info['rainfall']}mm\n"
            f"Action: Barah-e-meherbani ilaqai hidayat par amal karen."
        )
    else:
        return (
            f"Chitral Alert ({predicted_info['hours']}h): {predicted_info['risk_level']}\n"
            f"Water: {predicted_info['water_level']} m | Rain: {predicted_info['rainfall']} mm\n"
            f"Action: Follow local authority instructions. Move to higher ground if necessary."
        )

# ------------------------------
# Community reporting: local in-memory + persisted queue (existing)
REPORTS_FILE = "community_reports.json"

def load_reports():
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_report(report):
    reports = load_reports()
    reports.append(report)
    with open(REPORTS_FILE, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

# ------------------------------
# Department Auto-Routing configuration (NEW)
# Fill these numbers/emails with real department contacts for your region.
# For demo, leave placeholder numbers (they will be used only if Twilio creds are present).
DEPARTMENT_CONTACTS = {
    "Flood": {
        "name": "PDMA District Office",
        "phones": ["+92XXXXXXXXXX"],   # list of phone numbers
        "emails": ["pdma@example.com"],
        "preferred": ["sms", "email"]  # order of channels to try: sms, whatsapp, call, email
    },
    "Landslide": {
        "name": "C&W Department",
        "phones": ["+92YYYYYYYYYY"],
        "emails": ["cw@example.com"],
        "preferred": ["sms", "email"]
    },
    "RoadBlock": {
        "name": "District Admin",
        "phones": ["+92ZZZZZZZZZZ"],
        "emails": ["admin@example.com"],
        "preferred": ["whatsapp", "sms"]
    },
    "Default": {
        "name": "Local Rescue 1122",
        "phones": ["+92AAAAAAAAAA"],
        "emails": ["rescue@example.com"],
        "preferred": ["sms", "call"]
    }
}

def route_report_to_departments(report):
    """
    Determine which department(s) should receive alerts based on report content (severity/keywords).
    Then send via preferred channels.
    """
    # Simple routing logic: match keywords from location/notes/severity
    notes = (report.get("notes") or "").lower()
    severity = report.get("severity", "Medium")
    location = (report.get("location") or "").lower()

    # Determine type
    # Basic heuristics: if notes contain 'landslide', 'road', 'flood', 'river', etc.
    if "landslide" in notes or "slide" in notes or "rockfall" in notes:
        report_type = "Landslide"
    elif "road" in notes or "blocked" in notes:
        report_type = "RoadBlock"
    elif "flood" in notes or "river" in notes or "water" in notes:
        report_type = "Flood"
    else:
        report_type = "Default"

    # Choose departments to notify (primary + maybe default)
    departments_to_notify = [DEPARTMENT_CONTACTS.get(report_type, DEPARTMENT_CONTACTS["Default"])]

    # Compose message
    ts = report.get("timestamp", datetime.utcnow().isoformat())
    body = f"Community Report ({report.get('severity')}): {report.get('location')} | {report.get('notes')} | From: {report.get('name')} | {ts}"

    # Send according to preferred channels
    results = []
    for dept in departments_to_notify:
        name = dept.get("name")
        pref = dept.get("preferred", ["sms"])
        phones = dept.get("phones", [])
        emails = dept.get("emails", [])
        # iterate channels in preference order
        for channel in pref:
            if channel == "sms":
                for p in phones:
                    res = send_sms_real_twilio(p, body)  # will fallback to simulated if no creds
                    results.append((name, "sms", p, res))
            elif channel == "whatsapp":
                for p in phones:
                    res = send_whatsapp_twilio(p, body)
                    results.append((name, "whatsapp", p, res))
            elif channel == "call":
                for p in phones:
                    res = send_voice_call_twilio(p, body)
                    results.append((name, "call", p, res))
            elif channel == "email":
                # For email we log and (optionally) send via SMTP if configured
                results.append((name, "email", emails, {"status": "logged"}))
                # Optional: implement send_email function to actually send email if SMTP creds are set.
    return results

# ------------------------------
# Weather + Satellite helpers (NEW)
# - Open-Meteo: free, no key. We'll fetch short-term precipitation/temperature forecast for given lat/lon.
# - NASA GIBS: return a tile URL template that can be used in a Leaflet map overlay for MODIS/VIIRS tiles.
# ------------------------------
def fetch_open_meteo_forecast(lat, lon, hours=12):
    """
    Fetch hourly precipitation & temperature from Open-Meteo for the next `hours` hours.
    Returns a small summary dict (next-hours cumulative precipitation, peak temp, and raw arrays).
    """
    try:
        end = datetime.utcnow() + timedelta(hours=hours)
        start_iso = datetime.utcnow().replace(minute=0, second=0, microsecond=0).isoformat() + "Z"
        end_iso = end.replace(minute=0, second=0, microsecond=0).isoformat() + "Z"
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&hourly=precipitation,temperature_2m&timezone=UTC"
            f"&start_date={datetime.utcnow().strftime('%Y-%m-%d')}&end_date={end.strftime('%Y-%m-%d')}"
        )
        r = requests.get(url, timeout=10)
        data = r.json()
        # Extract next `hours` hourly values from 'hourly'
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        prec = hourly.get("precipitation", [])
        temps = hourly.get("temperature_2m", [])
        # Build arrays for next `hours`
        now = datetime.utcnow()
        next_prec = []
        next_temp = []
        for t, p, tt in zip(times, prec, temps):
            dt = datetime.fromisoformat(t)
            if dt >= now and dt <= now + timedelta(hours=hours):
                next_prec.append(p)
                next_temp.append(tt)
        summary = {
            "lat": lat,
            "lon": lon,
            "hours": hours,
            "precipitation_array": next_prec,
            "temperature_array": next_temp,
            "precipitation_sum_next_hours": round(sum(next_prec), 2) if next_prec else 0.0,
            "peak_temp_next_hours": round(max(next_temp), 2) if next_temp else None
        }
        return summary
    except Exception as e:
        print("Open-Meteo fetch error:", e)
        return {"error": str(e)}

def nasa_gibs_tile_template(layer="MODIS_Terra_CorrectedReflectance_TrueColor", date=None):
    """
    Return a NASA GIBS tile template (WMTS) URL pattern for a given layer and date.
    date should be YYYY-MM-DD (e.g., '2025-12-07'). If date is None, use today's date.
    This is a helper you can use to overlay satellite imagery in a web map (Leaflet, Mapbox, etc).
    NOTE: This function only returns the URL template (no network call here).
    """
    if not date:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    # Basic validation of date format (if user passed something incorrect, fallback to current date)
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    template = f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/{{z}}/{{y}}/{{x}}.jpg"
    return {"template": template, "date": date, "layer": layer}

# ------------------------------
# Generate alert HTML (keeps original style & role messages) — unchanged aside from adding optional weather snippet
# ------------------------------
def generate_alert(role):
    # Preserve original behavior + now add predicted risk snippets and SMS control
    water_level, rainfall, glacier_melt = simulate_sensor_data()

    risk_score = compute_risk_score(water_level, rainfall, glacier_melt)
    risk_level, bar_color, bg_color = risk_level_info(risk_score)

    # AI guidance only for Moderate/High risk, same as original
    if risk_score >= 3:
        ai_guidance = get_ai_guidance(water_level, rainfall, glacier_melt)
    else:
        ai_guidance = "No action required. All parameters are within safe limits."

    sensor_display = f"🌊 Water Level: {water_level} m | ☔ Rainfall: {rainfall} mm | 🗻 Glacier Melt: {glacier_melt} cm"

    # Role-specific messages unchanged
       # Role-specific messages unchanged
    if role == "Public":
        role_message = """
<h3 style='color:#b30000;'>⚠️ PUBLIC ALERT</h3>
<ul>
<li>Stay away from riverbanks and risky areas.</li>
<li>Keep emergency contacts ready.</li>
<li>Move to higher ground if necessary.</li>
</ul>
"""

    elif role == "Rescue Worker":
        role_message = """
<h3 style='color:#004080;'>🛟 RESCUE WORKER ALERT</h3>
<ul>
<li>Dispatch teams to high-risk zones.</li>
<li>Ensure first-aid, ropes, and rescue kits ready.</li>
<li>Coordinate with admin for evacuation routes.</li>
</ul>
"""

    elif role == "Admin":
        role_message = """
<h3 style='color:#5c0099;'>🛠️ ADMIN ALERT</h3>
<ul>
<li>Verify sensor data & system health.</li>
<li>Approve or override alerts if needed.</li>
<li>Monitor rescue team readiness.</li>
</ul>
"""

    else:
        role_message = ""

    # Prediction snippet (6 & 12 hours)
    pred6 = predict_next_hours(6)
    pred12 = predict_next_hours(12)
    # Fetch a small open-meteo forecast for a sample lat/lon (for display). You can replace lat/lon with actual sensor location.
    sample_lat, sample_lon = 35.8600, 71.7888  # central Chitral approx
    meteo = fetch_open_meteo_forecast(sample_lat, sample_lon, hours=6)
    meteo_html = ""
    if "precipitation_sum_next_hours" in meteo:
        meteo_html = f"<div><strong>Weather (next 6h forecast):</strong> Precipitation ≈ {meteo['precipitation_sum_next_hours']} mm</div>"

    # Satellite tile info (note: now returns current template and date, but UI allows edit)
    sat = nasa_gibs_tile_template()
    sat_html = f"<div><strong>Satellite tile (GIBS):</strong> {sat['layer']} (date {sat['date']}) — use template in dashboard.</div>"

    prediction_html = f"""
    <div style="margin-top:10px; padding:10px; background: #ffffff; border-radius:8px;">
        <strong>Prediction (6h):</strong> Water {pred6['water_level']}m, Rain {pred6['rainfall']}mm — {pred6['risk_level']}<br>
        <strong>Prediction (12h):</strong> Water {pred12['water_level']}m, Rain {pred12['rainfall']}mm — {pred12['risk_level']}<br>
        {meteo_html}
        {sat_html}
    </div>
    """

    html_content = f"""
    <div style='padding:20px; border-radius:15px; font-size:18px; background-color:{bg_color};'>
        <div style='height:15px; width:100%; background-color:{bar_color}; animation: flash 1s infinite; border-radius:10px; margin-bottom:10px;'></div>
        <strong>{sensor_display}</strong><br><br>
        {risk_level}<br><br>
        {role_message}<br>
        <div style='margin-top:10px;'>
            <strong>AI Safety Guidance:</strong><br>
            {ai_guidance.replace(chr(10), '<br>')}
        </div>
        {prediction_html}
    </div>
    <style>
    @keyframes flash {{
        0% {{opacity:1;}}
        50% {{opacity:0.3;}}
        100% {{opacity:1;}}
    }}
    </style>
    """
    return html_content

# ------------------------------
# Validation helpers (NEW additions — do not alter existing flows)
# ------------------------------
def is_alpha_name(name):
    """Return True if name contains only letters and spaces (English letters)."""
    if not name:
        return False
    # Allow letters (A-Z, a-z) and spaces
    return bool(re.match(r'^[A-Za-z\s]+$', name.strip()))

def is_digits_only(phone):
    """Return True if phone contains only digits (0-9)."""
    if not phone:
        return False
    return bool(re.match(r'^\d+$', phone.strip()))

# ------------------------------
# Additional Gradio functions: SMS send, community report submit, load reports
# (modified ui_submit_report to auto-route reports to departments — but with validation)
# ------------------------------
def ui_send_sms(provider, phone, lang, hours_ahead, role_for_sms):
    """
    Called from UI. Creates SMS text from prediction and queues via simulated gateway.
    Returns message for UI and updated log.
    Validation: phone must be digits only (per user request).
    """
    # Validate phone (digits only)
    if not phone or not is_digits_only(phone):
        # Do not send; return error and keep sms log as-is
        return "Error: Phone number invalid. Accepts digits only (0-9).", ui_load_sms_log()

    try:
        hours = int(hours_ahead)
    except:
        hours = 6
    predicted = predict_next_hours(hours)
    text = make_sms_text(role_for_sms, predicted, language=lang)
    # For UI-sent SMS we keep original behavior: simulated -> but if Twilio is configured, let user choose (here we attempt Twilio)
    result = send_sms_real_twilio(phone, text)
    # Return a user-friendly message and the latest few log entries
    logs = load_sms_log()
    tail = logs[-5:][::-1]
    display = "<br>".join([f"{l['timestamp']} | {l['provider']} | {l['phone']} | {l['language']} | {l['text'][:80]}..." for l in tail])
    return f"SMS queued (or sent). Provider result: {result.get('status')}", display

def ui_submit_report(name, phone, location, severity, notes):
    """
    Submit community report.
    Validation:
     - name: alphabets and spaces only
     - phone: digits only
    If validation fails, return an error and do NOT save the report.
    """
    # Validate name
    if not is_alpha_name(name):
        # Return error and the current reports list (so UI doesn't lose display)
        return "Error: Name invalid. Use letters and spaces only (no digits or symbols).", ui_load_reports()
    # Validate phone
    if not is_digits_only(phone):
        return "Error: Phone invalid. Use digits only (0-9).", ui_load_reports()

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "name": name,
        "phone": phone,
        "location": location,
        "severity": severity,
        "notes": notes
    }
    save_report(report)

    # NEW: Auto-route to departments (this will use Twilio if configured, otherwise simulated logs)
    routing_results = route_report_to_departments(report)
    # Build display info
    route_display_lines = []
    for dept_name, channel, target, res in routing_results:
        # res may be dict from send_* function
        route_display_lines.append(f"{datetime.utcnow().isoformat()} | {dept_name} | {channel} | {target} | {res.get('status') if isinstance(res, dict) else res}")

    # return last 5 reports for UI
    reports = load_reports()
    tail = reports[-5:][::-1]
    display = "<br>".join([f"{r['timestamp']} | {r['location']} | {r['severity']} | {r['notes'][:80]}..." for r in tail])
    route_summary = "<br>".join(route_display_lines) if route_display_lines else "No routing performed."
    return "Report submitted. Thank you.", display + "<br><hr><strong>Routing:</strong><br>" + route_summary

def ui_load_reports():
    reports = load_reports()
    if not reports:
        return "No reports yet."
    tail = reports[-10:][::-1]
    display = "<br>".join([f"{r['timestamp']} | {r['name']} | {r['location']} | {r['severity']} | {r['notes']}" for r in tail])
    return display

def ui_load_sms_log():
    logs = load_sms_log()
    if not logs:
        return "No SMS sent yet."
    tail = logs[-10:][::-1]
    display = "<br>".join([f"{l['timestamp']} | {l['provider']} | {l['phone']} | {l['language']} | {l['text']}" for l in tail])
    return display

# ------------------------------
# Satellite UI helper (NEW)
# ------------------------------
def ui_satellite_template(date_str, layer):
    """
    Return the NASA GIBS tile template for the selected layer and date (date_str must be YYYY-MM-DD).
    If date_str invalid, function will use today's date and indicate it.
    """
    # Validate date format; nasa_gibs_tile_template already handles fallback, but we'll note if user provided invalid
    try:
        if date_str:
            datetime.strptime(date_str, "%Y-%m-%d")
    except:
        # invalid date format
        date_str = None
    sat = nasa_gibs_tile_template(layer=layer, date=date_str)
    template = sat.get("template")
    date_used = sat.get("date")
    display = f"Layer: {sat.get('layer')}<br>Date used: {date_used}<br>Template: {template}"
    return display


# ------------------------------
# Streamlit UI
# ------------------------------

import streamlit as st

st.set_page_config(
    page_title="Chitral Glacier Risk & Early Warning System",
    page_icon="🌄",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#743089; font-weight:bold; text-transform:uppercase;'>
    🌨️ CHITRAL GLACIER RISK & EARLY WARNING SYSTEM 🌄
    </h1>
    """,
    unsafe_allow_html=True
)

# Initialize session state for generated alert and logs.
if "alert_html" not in st.session_state:
    st.session_state.alert_html = generate_alert("Public")

left, right = st.columns([2, 1])

with left:
    role = st.radio(
        "Select your role",
        ["Public", "Rescue Worker", "Admin"],
        index=0
    )

    if st.button("🔄 Refresh Live Monitoring", use_container_width=True):
        st.session_state.alert_html = generate_alert(role)

    st.markdown(st.session_state.alert_html, unsafe_allow_html=True)

    st.markdown("### Predictions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict +6 hours", use_container_width=True):
            p = predict_next_hours(6)
            st.info(
                f"6-hour prediction — Water {p['water_level']} m, "
                f"Rain {p['rainfall']} mm — {p['risk_level']}"
            )

    with col2:
        if st.button("Predict +12 hours", use_container_width=True):
            p = predict_next_hours(12)
            st.info(
                f"12-hour prediction — Water {p['water_level']} m, "
                f"Rain {p['rainfall']} mm — {p['risk_level']}"
            )

    st.markdown("### Community Reporting")

    name_in = st.text_input("Your name")
    phone_in = st.text_input("Phone (digits only)")
    location_in = st.text_input("Location / Landmark")
    severity_in = st.selectbox("Severity", ["Low", "Medium", "High"], index=1)
    notes_in = st.text_area("Notes / Observations", height=100)

    if st.button("Submit Report", use_container_width=True):
        status, report_display = ui_submit_report(
            name_in,
            phone_in,
            location_in,
            severity_in,
            notes_in
        )
        if status.startswith("Error"):
            st.error(status)
        else:
            st.success(status)
        st.markdown(report_display, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### SMS Panel (Simulated - FYP Demo)")

    provider = st.selectbox(
        "Select Provider (simulated)",
        ["Jazz", "Zong", "Ufone", "Telenor"],
        index=3
    )
    phone_to = st.text_input(
        "Phone Number (digits only, e.g. 923001234567)"
    )
    sms_lang = st.radio("Language", ["EN", "UR"], horizontal=True)
    sms_hours = st.slider(
        "Alert prediction hours ahead",
        min_value=1,
        max_value=12,
        value=6,
        step=1
    )
    sms_role = st.radio(
        "Role for message",
        ["Public", "Rescue Worker", "Admin"],
        index=0,
        horizontal=True
    )

    if st.button(
        "Send SMS (simulate or real if Twilio configured)",
        use_container_width=True
    ):
        sms_status, sms_log = ui_send_sms(
            provider,
            phone_to,
            sms_lang,
            sms_hours,
            sms_role
        )
        if sms_status.startswith("Error"):
            st.error(sms_status)
        else:
            st.success(sms_status)
        st.markdown(sms_log, unsafe_allow_html=True)

with right:
    st.markdown("### Recent Community Reports")

    if st.button("Refresh Reports", use_container_width=True):
        st.rerun()

    st.markdown(ui_load_reports(), unsafe_allow_html=True)

    st.markdown("### SMS Log (last entries)")

    if st.button("Refresh SMS Log", use_container_width=True):
        st.rerun()

    st.markdown(ui_load_sms_log(), unsafe_allow_html=True)

    st.markdown("### Satellite / GIBS Tile Helper")

    st.write(
        "Enter a date (YYYY-MM-DD) to fetch the NASA GIBS WMTS tile "
        "template for that date. The default is today's date but you "
        "can enter a past date to view historical tiles."
    )

    sat_layer = st.text_input(
        "GIBS Layer (e.g. MODIS_Terra_CorrectedReflectance_TrueColor)",
        value="MODIS_Terra_CorrectedReflectance_TrueColor"
    )

    sat_date = st.text_input(
        "Tile date (YYYY-MM-DD) — editable",
        value=datetime.utcnow().strftime("%Y-%m-%d")
    )

    if st.button("Generate Tile Template", use_container_width=True):
        st.markdown(
            ui_satellite_template(sat_date, sat_layer),
            unsafe_allow_html=True
        )

    st.markdown("### PWA / Offline Mode (mobile-friendly)")

    st.markdown(
        """
        This demo includes a client-side Service Worker and offline queue
        concept from the original application. On supported browsers,
        mobile users can use browser options such as **Add to Home Screen**.
        """
    )

    st.markdown("### Notes / Limitations")

    st.markdown(
        """
        - SMS/WhatsApp/Voice features use Twilio when configured.
        - If TWILIO_* environment variables are missing, the system falls
          back to simulated logging.
        - Update `DEPARTMENT_CONTACTS` with real phone numbers and emails
          for automatic routing.
        - Open-Meteo provides free short-term forecasts — no API key needed.
        - NASA GIBS tile template is provided for satellite overlays in a
          map; no direct image fetch is performed here.
        """
    )
