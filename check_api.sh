#!/bin/bash
KEY=$(grep KMA_API_KEY .env | cut -d= -f2 | tr -d "\r\n\"")
resp=$(curl -s --max-time 12 "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm=202608071000&stn=156&authKey=${KEY}")
echo "$resp" | grep -qa "^202608" && echo "✅ 회복됨 — 다시 수집 시작 가능" || echo "❌ 아직 막혀 있음"

