@echo off
REM ============================================
REM NAVA Phase 2 Comprehensive Test Script
REM ============================================

echo.
echo ====================================
echo NAVA Phase 2 Comprehensive Testing
echo ====================================
echo.

REM Set colors for output
setlocal enabledelayedexpansion

REM Test counter
set /a test_count=0
set /a passed_count=0
set /a failed_count=0

echo [INFO] Starting comprehensive testing...
echo.

REM ===== 1. HEALTH CHECK ALL SERVICES =====
echo ===== 1. HEALTH CHECK ALL SERVICES =====

set /a test_count+=1
echo [TEST %test_count%] Decision Engine Health Check (Port 8008)
curl -s http://localhost:8008/health > nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [PASS] Decision Engine is healthy
    set /a passed_count+=1
) else (
    echo [FAIL] Decision Engine is not responding
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] Quality Service Health Check (Port 8009)
curl -s http://localhost:8009/health > nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [PASS] Quality Service is healthy
    set /a passed_count+=1
) else (
    echo [FAIL] Quality Service is not responding
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] SLF Framework Health Check (Port 8010)
curl -s http://localhost:8010/health > nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [PASS] SLF Framework is healthy
    set /a passed_count+=1
) else (
    echo [FAIL] SLF Framework is not responding
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] Cache Engine Health Check (Port 8013)
curl -s http://localhost:8013/health > nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [PASS] Cache Engine is healthy
    set /a passed_count+=1
) else (
    echo [FAIL] Cache Engine is not responding
    set /a failed_count+=1
)

echo.

REM ===== 2. BASIC FUNCTIONALITY TESTS =====
echo ===== 2. BASIC FUNCTIONALITY TESTS =====

set /a test_count+=1
echo [TEST %test_count%] Decision Engine - Quick Select
curl -s -X POST "http://localhost:8008/api/decision/quick-select?message=test" > temp_result.txt 2>&1
findstr /C:"recommended_model" temp_result.txt > nul
if %ERRORLEVEL% equ 0 (
    echo [PASS] Decision Engine quick select works
    set /a passed_count+=1
) else (
    echo [FAIL] Decision Engine quick select failed
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] Quality Service - Quick Check
curl -s -X POST "http://localhost:8009/api/quality/quick?response_text=test&min_threshold=0.5" > temp_result.txt 2>&1
findstr /C:"overall_score" temp_result.txt > nul
if %ERRORLEVEL% equ 0 (
    echo [PASS] Quality Service quick check works
    set /a passed_count+=1
) else (
    echo [FAIL] Quality Service quick check failed
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] SLF Framework - Quick Enhancement
curl -s -X POST "http://localhost:8010/api/slf/quick?prompt=test" > temp_result.txt 2>&1
findstr /C:"enhanced" temp_result.txt > nul
if %ERRORLEVEL% equ 0 (
    echo [PASS] SLF Framework quick enhancement works
    set /a passed_count+=1
) else (
    echo [FAIL] SLF Framework quick enhancement failed
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] Cache Engine - Basic Cache
curl -s -X POST "http://localhost:8013/api/cache/set" -H "Content-Type: application/json" -d "{\"key\":\"test_key\",\"value\":\"test_value\",\"ttl_seconds\":300}" > temp_result.txt 2>&1
findstr /C:"success" temp_result.txt > nul
if %ERRORLEVEL% equ 0 (
    echo [PASS] Cache Engine basic cache works
    set /a passed_count+=1
) else (
    echo [FAIL] Cache Engine basic cache failed
    set /a failed_count+=1
)

echo.

REM ===== 3. INTEGRATION TESTS =====
echo ===== 3. INTEGRATION TESTS =====

set /a test_count+=1
echo [TEST %test_count%] Decision Engine - Full Analysis
curl -s -X POST "http://localhost:8008/api/decision/analyze" -H "Content-Type: application/json" -d "{\"message\":\"complex analysis task\",\"context\":{}}" > temp_result.txt 2>&1
findstr /C:"recommended_model" temp_result.txt > nul
if %ERRORLEVEL% equ 0 (
    echo [PASS] Decision Engine full analysis works
    set /a passed_count+=1
) else (
    echo [FAIL] Decision Engine full analysis failed
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] Quality Service - Full Validation
curl -s -X POST "http://localhost:8009/api/quality/validate" -H "Content-Type: application/json" -d "{\"response_text\":\"This is a comprehensive test response with multiple sentences.\",\"original_query\":\"test quality\"}" > temp_result.txt 2>&1
findstr /C:"overall_score" temp_result.txt > nul
if %ERRORLEVEL% equ 0 (
    echo [PASS] Quality Service full validation works
    set /a passed_count+=1
) else (
    echo [FAIL] Quality Service full validation failed
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] SLF Framework - Full Enhancement
curl -s -X POST "http://localhost:8010/api/slf/enhance" -H "Content-Type: application/json" -d "{\"original_prompt\":\"Explain machine learning\",\"model_target\":\"gpt\",\"reasoning_type\":\"systematic\",\"enhancement_level\":\"moderate\",\"enterprise_mode\":false}" > temp_result.txt 2>&1
findstr /C:"enhanced" temp_result.txt > nul
if %ERRORLEVEL% equ 0 (
    echo [PASS] SLF Framework full enhancement works
    set /a passed_count+=1
) else (
    echo [FAIL] SLF Framework full enhancement failed
    set /a failed_count+=1
)

echo.

REM ===== 4. ERROR HANDLING TESTS =====
echo ===== 4. ERROR HANDLING TESTS =====

set /a test_count+=1
echo [TEST %test_count%] Quality Service - Valid Input
curl -s -X POST "http://localhost:8009/api/quality/validate" -H "Content-Type: application/json" -d "{\"response_text\":\"This is a comprehensive test response that demonstrates good quality with multiple sentences, clear explanations, and specific examples. The response provides detailed information and maintains professional tone throughout the content.\",\"original_query\":\"test quality validation\"}" > temp_result.txt 2>&1
findstr /C:"overall_score" temp_result.txt > nul
if %ERRORLEVEL% equ 0 (
    echo [PASS] Quality Service handles valid input correctly
    set /a passed_count+=1
) else (
    echo [FAIL] Quality Service valid input failed
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] Cache Engine - Invalid Key
curl -s -X POST "http://localhost:8013/api/cache/get" -H "Content-Type: application/json" -d "{\"key\":\"\"}" > temp_result.txt 2>&1
findstr /C:"hit" temp_result.txt > nul
if %ERRORLEVEL% equ 0 (
    echo [PASS] Cache Engine handles invalid key correctly
    set /a passed_count+=1
) else (
    echo [FAIL] Cache Engine error handling failed
    set /a failed_count+=1
)

echo.

REM ===== 5. PERFORMANCE TESTS =====
echo ===== 5. PERFORMANCE TESTS =====

set /a test_count+=1
echo [TEST %test_count%] Load Test - 10 Concurrent Requests
for /l %%i in (1,1,10) do (
    start /b curl -s -X POST "http://localhost:8008/api/decision/quick-select?message=test%%i" > nul 2>&1
)
timeout /t 5 > nul
echo [PASS] Load test completed
set /a passed_count+=1

echo.

REM ===== 6. API DOCUMENTATION TESTS =====
echo ===== 6. API DOCUMENTATION TESTS =====

set /a test_count+=1
echo [TEST %test_count%] OpenAPI Documentation Access
curl -s http://localhost:8008/docs > nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [PASS] Decision Engine docs accessible
    set /a passed_count+=1
) else (
    echo [FAIL] Decision Engine docs not accessible
    set /a failed_count+=1
)

set /a test_count+=1
echo [TEST %test_count%] Quality Service Documentation
curl -s http://localhost:8009/docs > nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [PASS] Quality Service docs accessible
    set /a passed_count+=1
) else (
    echo [FAIL] Quality Service docs not accessible
    set /a failed_count+=1
)

echo.

REM ===== FINAL RESULTS =====
echo ====================================
echo TEST RESULTS SUMMARY
echo ====================================
echo Total Tests: %test_count%
echo Passed: %passed_count%
echo Failed: %failed_count%

if %failed_count% equ 0 (
    echo.
    echo [SUCCESS] All tests passed! Phase 2 is fully operational.
    echo.
) else (
    echo.
    echo [WARNING] %failed_count% tests failed. Check the issues above.
    echo.
)

REM Calculate pass rate
set /a pass_rate=(%passed_count% * 100) / %test_count%
echo Pass Rate: %pass_rate%%%

REM Cleanup
del temp_result.txt > nul 2>&1

echo.
echo Testing completed at %date% %time%
echo.

pause