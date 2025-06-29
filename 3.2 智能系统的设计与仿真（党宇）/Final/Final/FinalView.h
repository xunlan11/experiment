// FinalView.h: CFinalView 类的接口
//

#pragma once
#include <vector>  // 添加vector头文件

using namespace std;  // 添加标准命名空间

#include "BasicDlg.h"
#include "ParameterDlg.h"
#include "SimulationDlg.h"

// 绘图数据结构
struct PlotData
{
	vector<double> time;      // 时间点
	vector<double> torque;    // 控制扭矩
	vector<double> error;     // 误差
	vector<double> angle;     // 实际角度
	vector<double> refAngle;  // 参考角度
};

class CFinalView : public CView
{
protected: // 仅从序列化创建
	CFinalView() noexcept;
	DECLARE_DYNCREATE(CFinalView)

// 特性
public:
	CFinalDoc* GetDocument() const;

// 操作
public:
	// 仿真参数
	double m_dJ;            // 惯性矩
	double m_dB;            // 阻尼系数
	double m_dA;            // 固定摩擦系数
	double m_dTauG;         // 重力扭矩
	double m_dInitialAngle; // 初始角度(弧度)
	
	double m_dSimTime;      // 仿真时间
	double m_dTimeStep;     // 仿真步长
	bool m_bConstantAngle;  // 目标角度类型
	double m_dTargetAngle;  // 目标角度幅值(弧度)
	double m_dSinePeriod;   // 正弦曲线周期
	bool m_bRK4;            // 是否使用四阶龙格库塔法
	
	double m_dKp;           // 比例系数
	double m_dKi;           // 积分系数
	double m_dKd;           // 微分系数
	
	// 存储PID参数记录，使其在应用程序运行期间持久存在
	CArray<PIDParams, PIDParams&> m_arrPIDParams;  // PID参数数组
	
	// 图形设置
	CString m_strFontName;
	int m_nFontSize;
	int m_nXInterval;
	int m_nAxisWidth;
	COLORREF m_colorAxis;
	
	// 仿真状态
	bool m_bSimulating;     // 是否正在仿真
	bool m_bPaused;         // 是否暂停
	bool m_bSimulationCompleted; // Flag to track if simulation has completed normally
	double m_dCurrentTime;  // 当前仿真时间
	double m_dCurrentAngle; // 当前角度(弧度)
	double m_dCurrentVelocity; // 当前角速度
	double m_dIntegralError;   // 积分误差
	double m_dLastError;       // 上一次误差
	
	// 绘图数据
	PlotData m_plotData;
	
	// 双缓冲绘图相关
	CDC m_memDC;           // 内存DC
	CBitmap m_memBitmap;   // 内存位图
	CBitmap* m_pOldBitmap; // 旧位图
	CRect m_rectClient;    // 客户区矩形
	bool m_bBufferCreated; // 缓冲区是否已创建
	
	// 重写以绘制视图
	void DrawPlots(CDC* pDC, CRect& rectClient);
	void DrawKneeAnimation(CDC* pDC, CRect& rectClient);
	void DrawParameterInfo(CDC* pDC, CRect& rectClient); // 添加参数信息绘制函数
	double GetTargetAngle(double time);
	double GetControlTorque(double currentAngle, double targetAngle);
	
	// 欧拉法更新系统状态
	void EulerUpdate();
	
	// 四阶龙格库塔法更新系统状态
	void RK4Update();
	
	// 系统动力学方程
	void SystemDynamics(double theta, double omega, double tau, double& dtheta, double& domega);
	
	// 保存和加载设置
	void SaveSettings(LPCTSTR lpszFileName);
	void LoadSettings(LPCTSTR lpszFileName);

// 重写
public:
	virtual void OnDraw(CDC* pDC);  // 重写以绘制该视图
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnInitialUpdate();

// 实现
public:
	virtual ~CFinalView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:
	// 初始化参数
	void InitParams();
	
	// 初始化仿真
	void InitSimulation();
	
	// 停止仿真
	void StopSimulation();
	
	// 创建和销毁双缓冲
	void CreateBuffer(CDC* pDC);
	void DeleteBuffer();

// 生成的消息映射函数
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnMenuBasic();
	afx_msg void OnMenuParameter();
	afx_msg void OnMenuSimulation();
	afx_msg void OnMenuSave();
	afx_msg void OnMenuLoad();
	afx_msg void OnButtonStart();
	afx_msg void OnButtonPause();
	afx_msg void OnButtonStop();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnSize(UINT nType, int cx, int cy);
protected:
	afx_msg LRESULT OnShowBasic(WPARAM wParam, LPARAM lParam);
};

#ifndef _DEBUG  // FinalView.cpp 中的调试版本
inline CFinalDoc* CFinalView::GetDocument() const
   { return reinterpret_cast<CFinalDoc*>(m_pDocument); }
#endif