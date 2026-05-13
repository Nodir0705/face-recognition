# PIPA compliance notes (개인정보보호법)

**This is a practical engineering checklist, not legal advice.** Before deploying, have HR or a 개인정보보호 컨설턴트 review your setup. The Personal Information Protection Commission (PIPC, 개인정보보호위원회) publishes guidance specifically for biometric attendance systems — read the latest version.

## What kind of data this is

Face embeddings used for identification are **biometric data** (생체정보) under PIPA Article 23. They are classed as **sensitive information** (민감정보), which gets stricter treatment than ordinary personal data.

## Minimum compliance checklist

### 1. Separate, explicit consent

Get **written consent** from every employee, in a form separate from the general employment contract. The consent form must state:

- The purpose: 출근/퇴근 기록 (attendance recording)
- The data items collected: 얼굴 이미지에서 추출한 특징 벡터 (face embedding), 출근/퇴근 시각, 출근/퇴근 유형
- The retention period: e.g., 재직 기간 + 3년 (employment + 3 years), or whatever your labor lawyer recommends
- The right to refuse and the alternative (e.g., RFID card, manual log)
- The right to withdraw consent at any time

If an employee refuses, you must offer an equivalent attendance method — refusal cannot be grounds for adverse treatment.

### 2. Provide an alternative

Have a working non-biometric fallback. The admin web UI's "manual entry" feature serves this purpose — employees who don't consent can have their attendance logged by a manager.

### 3. Data minimization

This system is designed with minimization in mind:

- **No raw photos stored** by default (`privacy.delete_photos_after_enrollment: true` in config.yaml). Only the 512-dim embedding remains.
- **Embeddings are non-invertible.** Unlike photos, you cannot reconstruct a face from a 512-float vector.
- **No video recording.** The camera stream is processed in memory and discarded.
- **No data leaves the LAN** except the structured log rows (employee ID, name, timestamp, in/out) sent to Google Sheets. Embeddings stay on the Pi.

Document this in your **개인정보 처리방침** (privacy policy).

### 4. Access control

- Change the default admin password in `scripts/attendance-admin.service` before deployment.
- The admin UI should be reachable only from the office network — do not port-forward.
- The Pi should not be accessible from the public internet. SSH should be key-only.
- Restrict who can view the Google Sheet (Share settings → specific people only).

### 5. Retention and deletion

- Configure `privacy.local_retention_days` in `config.yaml` to match your retention policy.
- Add a cron job to purge logs older than the retention window (script not included — write to suit your policy).
- On employee departure, deactivate them in the admin UI; on the right-to-erasure request, fully delete via the SQL commands in `enrollment.md`.

### 6. Logging access

PIPA requires you log who accessed sensitive data and when. Suggestions:

- Enable systemd journal retention (`/etc/systemd/journald.conf`: `Storage=persistent`, `MaxRetentionSec=1year`).
- Log all admin UI access (add a simple access-log middleware to `admin_web.py` — not included by default).
- Restrict SSH and `sqlite3` access on the Pi to one named admin account.

### 7. Notification

Post a notice at the kiosk in Korean and (if relevant) English:

> 이 장치는 출근/퇴근 기록을 위해 얼굴 인식을 사용합니다.
> 사전 동의서를 제출한 직원만 인식 대상이며, 동의하지 않은 직원은 [수동 기록 방법]을 이용하실 수 있습니다.
> 문의: HR (xxx-xxxx)

### 8. Data breach plan

If the Pi is lost or the SD card is stolen:

- Embeddings are on the device. While they can't be reversed into photos, they could be used to match against other images of the same people — treat the device as a sensitive asset.
- Encrypt the SD card (Bookworm supports LUKS) for additional protection.
- Have a documented breach-notification plan per PIPA Article 34.

## What this system explicitly does NOT do

To stay within "attendance recording" scope:

- ❌ Continuous video recording or surveillance
- ❌ Emotion/behavior analysis
- ❌ Identification of non-employees (visitors, delivery workers)
- ❌ Sharing data with third parties beyond Google (where Sheets is hosted)
- ❌ Cross-matching against external face databases

If anyone asks for any of the above, that's a different system with much stricter compliance requirements (and probably additional consent).
