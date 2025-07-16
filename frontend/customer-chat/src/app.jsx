// frontend/customer-chat/src/App.jsx
import React, { useState, useEffect } from 'react';
import Chat from './components/Chat';      // คอมโพเนนต์หน้าแชทที่เราสร้างไว้
import api from './services/api';        // API Service สำหรับติดต่อ Backend
import './styles/main.css';             // สไตล์ชีตหลักของแอป

function App() {
  // --- State Management ---
  // ใช้ useState เพื่อเก็บสถานะของ API แทนการเก็บใน class
  // สถานะเริ่มต้นคือ 'checking'
  const [apiHealth, setApiHealth] = useState({
    status: 'checking', // 'checking', 'healthy', 'unhealthy'
    message: 'Connecting to server...'
  });

  // --- Side Effects ---
  // ใช้ useEffect เพื่อจัดการ Logic ที่ต้องทำงานตอนเริ่มต้นและทำงานต่อเนื่อง
  // เทียบเท่ากับ init() และ setInterval() ในโค้ดเดิม
  useEffect(() => {
    console.log('🚀 NAVA Chat App starting...');

    // สร้างฟังก์ชันสำหรับเช็คสถานะเพื่อเรียกใช้ซ้ำ
    const performHealthCheck = async () => {
      try {
        const health = await api.checkHealth(); // เรียก API
        if (health.status === 'healthy') {
          setApiHealth({
            status: 'healthy',
            message: health.database === 'connected' ? 'Connected • Database OK' : 'Connected'
          });
        } else {
          // กรณี API ตอบกลับว่าไม่ healthy
          throw new Error('API reported an unhealthy status.');
        }
      } catch (error) {
        console.error('Health check failed:', error);
        setApiHealth({
          status: 'unhealthy',
          message: 'Disconnected'
        });
      }
    };

    // 1. เรียกเช็คสถานะครั้งแรกทันทีเมื่อคอมโพเนนต์ถูกโหลด
    performHealthCheck();

    // 2. ตั้งค่าให้มีการเช็คสถานะทุกๆ 30 วินาที
    const intervalId = setInterval(performHealthCheck, 30000);

    // 3. (สำคัญมาก) Cleanup function: จะทำงานเมื่อคอมโพเนนต์ถูกปิด
    // เพื่อยกเลิกการทำงานของ setInterval ป้องกัน memory leak
    return () => {
      console.log('🧹 Clearing health check interval.');
      clearInterval(intervalId);
    };

  }, []); // dependency array ว่าง [] หมายถึงให้ useEffect นี้ทำงานแค่ครั้งเดียวตอนเริ่มต้น

  // --- Render ---
  // ส่วนของการ "วาด" UI ทั้งหมด
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>NAVA Customer Assistant</h1>
        {/* แสดงสถานะที่ดึงมาจาก State โดยตรง */}
        <div id="statusIndicator" className="status-indicator">
          <span 
            className="status-dot" 
            style={{ 
              backgroundColor: apiHealth.status === 'healthy' ? '#28a745' : (apiHealth.status === 'unhealthy' ? '#dc3545' : '#ffc107') 
            }}
          ></span>
          <span>{apiHealth.message}</span>
        </div>
      </header>

      <main className="main-content">
        {/* แสดงผลคอมโพเนนต์แชท */}
        <Chat />
      </main>
    </div>
  );
}

export default App;