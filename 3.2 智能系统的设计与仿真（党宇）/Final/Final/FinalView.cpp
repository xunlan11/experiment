// FinalView.cpp: CFinalView 类的实现
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "Final.h"
#endif

#include "FinalDoc.h"
#include "FinalView.h"
#include "BasicDlg.h"
#include "ParameterDlg.h"
#include "SimulationDlg.h"
#include <cmath>
#include <vector> 
#include <algorithm>
#include <limits>

using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// 添加DBL_MAX定义
#ifndef DBL_MAX
#define DBL_MAX 1.7976931348623158e+308
#endif

#define PI 3.14159265358979323846
#define DEG2RAD(deg) ((deg) * PI / 180.0)
#define RAD2DEG(rad) ((rad) * 180.0 / PI)

// 在开始的宏定义部分添加自定义消息ID
#define WM_SHOW_BASIC (WM_USER + 100)

// CFinalView

IMPLEMENT_DYNCREATE(CFinalView, CView)

BEGIN_MESSAGE_MAP(CFinalView, CView)
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_COMMAND(ID_MENU_BASIC, &CFinalView::OnMenuBasic)
	ON_COMMAND(ID_MENU_PARAMETER, &CFinalView::OnMenuParameter)
	ON_COMMAND(ID_MENU_SIMULATION, &CFinalView::OnMenuSimulation)
	ON_COMMAND(ID_MENU_SAVE, &CFinalView::OnMenuSave)
	ON_COMMAND(ID_MENU_LOAD, &CFinalView::OnMenuLoad)
	ON_COMMAND(ID_BUTTON_START, &CFinalView::OnButtonStart)
	ON_COMMAND(ID_BUTTON_PAUSE, &CFinalView::OnButtonPause)
	ON_COMMAND(ID_BUTTON_STOP, &CFinalView::OnButtonStop)
	ON_WM_TIMER()
	ON_WM_SIZE()
	ON_MESSAGE(WM_SHOW_BASIC, &CFinalView::OnShowBasic)
END_MESSAGE_MAP()

// CFinalView 构造/析构

CFinalView::CFinalView() noexcept
{
	// 初始化参数
	InitParams();
	// 初始化仿真状态
	m_bSimulating = false;
	m_bPaused = false;
	m_bSimulationCompleted = false; 
	m_dCurrentTime = 0.0;
	m_dCurrentAngle = m_dInitialAngle;
	m_dCurrentVelocity = 0.0;
	m_dIntegralError = 0.0;
	m_dLastError = 0.0;
	// 初始化绘图数据
	m_plotData.time.clear();
	m_plotData.torque.clear();
	m_plotData.error.clear();
	m_plotData.angle.clear();
	m_plotData.refAngle.clear();
	// 初始化PID参数记录数组
	m_arrPIDParams.RemoveAll();
	// 初始化双缓冲
	m_bBufferCreated = false;
	m_pOldBitmap = NULL;
}

CFinalView::~CFinalView()
{
	// 释放双缓冲绘图资源
	DeleteBuffer();
}

// 初始化参数
void CFinalView::InitParams()
{
	// 系统参数默认值
	m_dJ = 0.5;            // 惯性矩
	m_dB = 0.5;            // 阻尼系数
	m_dA = 0.2;            // 固定摩擦系数
	m_dTauG = 10.0;        // 重力扭矩
	m_dInitialAngle = 0.0; // 初始角度
	// 仿真参数默认值
	m_dSimTime = 10.0;      // 仿真时间
	m_dTimeStep = 0.01;     // 仿真步长
	m_bConstantAngle = true; // 目标角度类型(定值)
	m_dTargetAngle = DEG2RAD(30.0); // 目标角度幅值
	m_dSinePeriod = 2.0;    // 正弦曲线周期
	m_bRK4 = false;         // 默认欧拉法
	// PID参数默认值
	m_dKp = 10.0;
	m_dKi = 5.0;
	m_dKd = 2.0;
	// 图形设置默认值
	m_strFontName = _T("Arial");
	m_nFontSize = 16;     
	m_nXInterval = 1;
	m_nAxisWidth = 2;      
	m_colorAxis = RGB(0, 0, 0);
}

// 初始化仿真
void CFinalView::InitSimulation()
{
	// 清除之前的数据
	m_plotData.time.clear();
	m_plotData.torque.clear();
	m_plotData.error.clear();
	m_plotData.angle.clear();
	m_plotData.refAngle.clear();
	// 初始化状态
	m_dCurrentTime = 0.0;
	m_dCurrentAngle = DEG2RAD(m_dInitialAngle);
	m_dCurrentVelocity = 0.0;
	m_dIntegralError = 0.0;
	m_dLastError = 0.0;
	m_bSimulationCompleted = false;
	// 添加初始数据点
	double targetAngle = GetTargetAngle(m_dCurrentTime);
	m_plotData.time.push_back(m_dCurrentTime);
	m_plotData.torque.push_back(0.0);
	m_plotData.error.push_back(targetAngle - m_dCurrentAngle);
	m_plotData.angle.push_back(m_dCurrentAngle);
	m_plotData.refAngle.push_back(targetAngle);
	// 设置仿真状态
	m_bSimulating = true;
	m_bPaused = false;
	// 启动定时器，增加定时器间隔为100毫秒，减少刷新频率
	SetTimer(ID_TIMER_SIMULATION, 100, NULL);
}

// 结束仿真
void CFinalView::StopSimulation()
{
	// 停止定时器
	KillTimer(ID_TIMER_SIMULATION);
	if (!m_bSimulationCompleted)
	{
		// 清除绘图数据
		m_plotData.time.clear();
		m_plotData.torque.clear();
		m_plotData.error.clear();
		m_plotData.angle.clear();
		m_plotData.refAngle.clear();
		// 重置当前状态
		m_dCurrentTime = 0.0;
		m_dCurrentAngle = m_dInitialAngle;
		m_dCurrentVelocity = 0.0;
		m_dIntegralError = 0.0;
		m_dLastError = 0.0;
	}
	// 设置仿真状态
	m_bSimulating = false;
	m_bPaused = false;
}

BOOL CFinalView::PreCreateWindow(CREATESTRUCT& cs)
{
	return CView::PreCreateWindow(cs);
}

// CFinalView 绘图
void CFinalView::OnDraw(CDC* pDC)
{
	CFinalDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;
	// 确保缓冲区已创建
	if (!m_bBufferCreated)
	{
		CreateBuffer(pDC);
	}
	// 清空缓冲区
	m_memDC.FillSolidRect(m_rectClient, RGB(255, 255, 255));
	// 绘制曲线图和动画到内存DC
	DrawPlots(&m_memDC, m_rectClient);
	DrawParameterInfo(&m_memDC, m_rectClient); // 添加参数信息绘制
	DrawKneeAnimation(&m_memDC, m_rectClient);
	// 将内存DC的内容复制到屏幕
	pDC->BitBlt(0, 0, m_rectClient.Width(), m_rectClient.Height(), &m_memDC, 0, 0, SRCCOPY);
}

// 绘制曲线图
void CFinalView::DrawPlots(CDC* pDC, CRect& rectClient)
{
	// 计算曲线图区域(左半部分)
	CRect rectPlot = rectClient;
	rectPlot.right = rectClient.right / 2;
	// 三个图表平均分配高度，各占三分之一
	int height = rectPlot.Height() / 3;
	// 扭矩曲线区域
	CRect rectTorque = rectPlot;
	rectTorque.bottom = rectPlot.top + height;
	// 误差曲线区域
	CRect rectError = rectPlot;
	rectError.top = rectTorque.bottom;
	rectError.bottom = rectError.top + height;
	// 角度曲线区域
	CRect rectAngle = rectPlot;
	rectAngle.top = rectError.bottom;
	// 创建字体
	CFont font;
	font.CreateFont(m_nFontSize + 4, 0, 0, 0, FW_NORMAL, FALSE, FALSE, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, m_strFontName);
	// 保存原字体
	CFont* pOldFont = pDC->SelectObject(&font);
	// 创建画笔
	CPen penAxis(PS_SOLID, m_nAxisWidth, m_colorAxis);
	CPen penGrid(PS_DOT, 1, RGB(200, 200, 200));
	CPen penTorque(PS_SOLID, 2, RGB(255, 0, 0));
	CPen penError(PS_SOLID, 2, RGB(0, 0, 255));
	CPen penAngle(PS_SOLID, 2, RGB(0, 128, 0));
	CPen penRefAngle(PS_SOLID, 2, RGB(255, 0, 255));
	// 保存原画笔
	CPen* pOldPen = pDC->SelectObject(&penAxis);
	// 绘制扭矩曲线标题和框架
	pDC->TextOut(rectTorque.left + 10, rectTorque.top + 10, _T("控制扭矩曲线"));
	// 绘制误差曲线标题和框架
	pDC->TextOut(rectError.left + 10, rectError.top + 10, _T("误差曲线"));
	// 绘制角度曲线标题和框架
	pDC->TextOut(rectAngle.left + 10, rectAngle.top + 10, _T("膝关节角度曲线"));
	// 设置绘图区域边距
	int margin = 50;
	// 扭矩曲线
	CRect rectTorquePlot = rectTorque;
	rectTorquePlot.DeflateRect(margin, margin, margin, margin);
	// 误差曲线
	CRect rectErrorPlot = rectError;
	rectErrorPlot.DeflateRect(margin, margin, margin, margin);
	// 角度曲线
	CRect rectAnglePlot = rectAngle;
	rectAnglePlot.DeflateRect(margin, margin, margin, margin);
	// 设置默认值范围
	double minTorque = -50.0, maxTorque = 50.0;
	double minError = -50.0, maxError = 50.0;
	double minAngle = DEG2RAD(-90.0), maxAngle = DEG2RAD(90.0);
	double displayTime = m_dSimTime;
	// 如果有数据，使用实际数据范围
	if (m_plotData.time.size() > 0)
	{
		// 找出数据的最大值和最小值
		maxTorque = -DBL_MAX, minTorque = DBL_MAX;
		maxError = -DBL_MAX, minError = DBL_MAX;
		maxAngle = -DBL_MAX, minAngle = DBL_MAX;
		for (size_t i = 0; i < m_plotData.time.size(); i++)
		{
			if (m_plotData.torque[i] > maxTorque) maxTorque = m_plotData.torque[i];
			if (m_plotData.torque[i] < minTorque) minTorque = m_plotData.torque[i];
			if (m_plotData.error[i] > maxError) maxError = m_plotData.error[i];
			if (m_plotData.error[i] < minError) minError = m_plotData.error[i];
			if (m_plotData.angle[i] > maxAngle) maxAngle = m_plotData.angle[i];
			if (m_plotData.angle[i] < minAngle) minAngle = m_plotData.angle[i];
			if (m_plotData.refAngle[i] > maxAngle) maxAngle = m_plotData.refAngle[i];
			if (m_plotData.refAngle[i] < minAngle) minAngle = m_plotData.refAngle[i];
		}
		// 确保有一定的边距
		double torqueRange = maxTorque - minTorque;
		double errorRange = maxError - minError;
		double angleRange = maxAngle - minAngle;
		if (torqueRange < 0.1) torqueRange = 0.1;
		if (errorRange < 0.1) errorRange = 0.1;
		if (angleRange < 0.1) angleRange = 0.1;
		minTorque -= torqueRange * 0.1;
		maxTorque += torqueRange * 0.1;
		minError -= errorRange * 0.1;
		maxError += errorRange * 0.1;
		minAngle -= angleRange * 0.1;
		maxAngle += angleRange * 0.1;
	}
	// 每个图绘制6个刻度点
	int yTickCount = 6;
	// 根据X轴标注间隔计算时间刻度点
	int timeTickCount = (int)(displayTime / m_nXInterval) + 1; // 根据间隔计算刻度点数量
	// 确保刻度点数量合理
	if (timeTickCount < 2) timeTickCount = 2; // 至少有起点和终点
	if (timeTickCount > 20) timeTickCount = 20; // 限制最大刻度点数，防止过密
	// 绘制网格线
	pDC->SelectObject(&penGrid);
	for (int i = 0; i < yTickCount; i++)
	{
		// 扭矩曲线水平网格线
		int y = rectTorquePlot.bottom - (int)(rectTorquePlot.Height() * i / (yTickCount - 1));
		pDC->MoveTo(rectTorquePlot.left, y);
		pDC->LineTo(rectTorquePlot.right, y);
		// 误差曲线水平网格线
		y = rectErrorPlot.bottom - (int)(rectErrorPlot.Height() * i / (yTickCount - 1));
		pDC->MoveTo(rectErrorPlot.left, y);
		pDC->LineTo(rectErrorPlot.right, y);
		// 角度曲线水平网格线
		y = rectAnglePlot.bottom - (int)(rectAnglePlot.Height() * i / (yTickCount - 1));
		pDC->MoveTo(rectAnglePlot.left, y);
		pDC->LineTo(rectAnglePlot.right, y);
	}
	for (int i = 0; i < timeTickCount; i++)
	{
		// 根据X轴间隔计算时间点
		double t = i * m_nXInterval;
		if (i == timeTickCount - 1) // 确保最后一个点是仿真时间
			t = displayTime;
		// 计算对应的像素位置
		int x = rectTorquePlot.left + (int)(rectTorquePlot.Width() * t / displayTime);
		// 扭矩曲线垂直网格线
		pDC->MoveTo(x, rectTorquePlot.top);
		pDC->LineTo(x, rectTorquePlot.bottom);
		// 误差曲线垂直网格线
		pDC->MoveTo(x, rectErrorPlot.top);
		pDC->LineTo(x, rectErrorPlot.bottom);
		// 角度曲线垂直网格线
		pDC->MoveTo(x, rectAnglePlot.top);
		pDC->LineTo(x, rectAnglePlot.bottom);
	}
	// 绘制坐标轴线
	pDC->SelectObject(&penAxis);
	// 绘制扭矩曲线坐标轴线
	// 绘制Y轴线
	pDC->MoveTo(rectTorquePlot.left, rectTorquePlot.top);
	pDC->LineTo(rectTorquePlot.left, rectTorquePlot.bottom);
	// 绘制X轴线
	pDC->MoveTo(rectTorquePlot.left, rectTorquePlot.bottom);
	pDC->LineTo(rectTorquePlot.right, rectTorquePlot.bottom);
	// 绘制误差曲线坐标轴线
	// 绘制Y轴线
	pDC->MoveTo(rectErrorPlot.left, rectErrorPlot.top);
	pDC->LineTo(rectErrorPlot.left, rectErrorPlot.bottom);
	// 绘制X轴线
	pDC->MoveTo(rectErrorPlot.left, rectErrorPlot.bottom);
	pDC->LineTo(rectErrorPlot.right, rectErrorPlot.bottom);
	// 绘制角度曲线坐标轴线
	// 绘制Y轴线
	pDC->MoveTo(rectAnglePlot.left, rectAnglePlot.top);
	pDC->LineTo(rectAnglePlot.left, rectAnglePlot.bottom);
	// 绘制X轴线
	pDC->MoveTo(rectAnglePlot.left, rectAnglePlot.bottom);
	pDC->LineTo(rectAnglePlot.right, rectAnglePlot.bottom);
	// 扭矩曲线Y轴刻度
	for (int i = 0; i < yTickCount; i++)
	{
		double value = minTorque + (maxTorque - minTorque) * i / (yTickCount - 1);
		int y = rectTorquePlot.bottom - (int)(rectTorquePlot.Height() * i / (yTickCount - 1));
		// 绘制刻度线
		pDC->MoveTo(rectTorquePlot.left - 5, y);
		pDC->LineTo(rectTorquePlot.left, y);
		// 显示刻度值
		CString str;
		str.Format(_T("%.2f"), value);
		pDC->TextOut(rectTorquePlot.left - 45, y - 7, str);
	}
	// 误差曲线Y轴刻度
	for (int i = 0; i < yTickCount; i++)
	{
		double value = minError + (maxError - minError) * i / (yTickCount - 1);
		int y = rectErrorPlot.bottom - (int)(rectErrorPlot.Height() * i / (yTickCount - 1));
		// 绘制刻度线
		pDC->MoveTo(rectErrorPlot.left - 5, y);
		pDC->LineTo(rectErrorPlot.left, y);
		// 显示刻度值
		CString str;
		str.Format(_T("%.2f"), value);
		pDC->TextOut(rectErrorPlot.left - 45, y - 7, str);
	}
	// 角度曲线Y轴刻度
	for (int i = 0; i < yTickCount; i++)
	{
		double value = minAngle + (maxAngle - minAngle) * i / (yTickCount - 1);
		int y = rectAnglePlot.bottom - (int)(rectAnglePlot.Height() * i / (yTickCount - 1));
		// 绘制刻度线
		pDC->MoveTo(rectAnglePlot.left - 5, y);
		pDC->LineTo(rectAnglePlot.left, y);
		// 显示刻度值
		CString str;
		str.Format(_T("%.1f°"), RAD2DEG(value));
		pDC->TextOut(rectAnglePlot.left - 45, y - 7, str);
	}
	// 绘制X轴刻度
	for (int i = 0; i < timeTickCount; i++)
	{
		// 根据X轴间隔计算时间点
		double t = i * m_nXInterval;
		if (i == timeTickCount - 1) // 确保最后一个点是仿真时间
			t = displayTime;
		// 计算对应的像素位置
		int x = rectTorquePlot.left + (int)(rectTorquePlot.Width() * t / displayTime);
		// 绘制刻度线
		pDC->SelectObject(&penAxis);
		// 扭矩曲线X轴刻度线
		pDC->MoveTo(x, rectTorquePlot.bottom);
		pDC->LineTo(x, rectTorquePlot.bottom + 5);
		// 误差曲线X轴刻度线
		pDC->MoveTo(x, rectErrorPlot.bottom);
		pDC->LineTo(x, rectErrorPlot.bottom + 5);
		// 角度曲线X轴刻度线
		pDC->MoveTo(x, rectAnglePlot.bottom);
		pDC->LineTo(x, rectAnglePlot.bottom + 5);
		// 显示刻度值
		CString str;
		str.Format(_T("%.1fs"), t);
		pDC->TextOut(x - 10, rectTorquePlot.bottom + 5, str);
		pDC->TextOut(x - 10, rectErrorPlot.bottom + 5, str);
		pDC->TextOut(x - 10, rectAnglePlot.bottom + 5, str);
	}
	// 只在有数据时绘制曲线
	if (m_plotData.time.size() > 0)
	{
		// 绘制扭矩曲线
		pDC->SelectObject(&penTorque);
		for (size_t i = 1; i < m_plotData.time.size(); i++)
		{
			// 只绘制displayTime范围内的数据
			if (m_plotData.time[i-1] > displayTime)
				break;
			int x1 = rectTorquePlot.left + (int)(rectTorquePlot.Width() * m_plotData.time[i-1] / displayTime);
			int y1 = rectTorquePlot.bottom - (int)(rectTorquePlot.Height() * (m_plotData.torque[i-1] - minTorque) / (maxTorque - minTorque));
			// 确保第二个点在显示范围内
			double time2 = min(m_plotData.time[i], displayTime);
			int x2 = rectTorquePlot.left + (int)(rectTorquePlot.Width() * time2 / displayTime);
			int y2 = rectTorquePlot.bottom - (int)(rectTorquePlot.Height() * (m_plotData.torque[i] - minTorque) / (maxTorque - minTorque));
			pDC->MoveTo(x1, y1);
			pDC->LineTo(x2, y2);
			// 如果超出显示范围，停止绘制
			if (m_plotData.time[i] >= displayTime)
				break;
		}
		// 误差曲线
		pDC->SelectObject(&penError);
		for (size_t i = 1; i < m_plotData.time.size(); i++)
		{
			// 只绘制displayTime范围内的数据
			if (m_plotData.time[i-1] > displayTime)
				break;
			int x1 = rectErrorPlot.left + (int)(rectErrorPlot.Width() * m_plotData.time[i-1] / displayTime);
			int y1 = rectErrorPlot.bottom - (int)(rectErrorPlot.Height() * (m_plotData.error[i-1] - minError) / (maxError - minError));
			// 确保第二个点在显示范围内
			double time2 = min(m_plotData.time[i], displayTime);
			int x2 = rectErrorPlot.left + (int)(rectErrorPlot.Width() * time2 / displayTime);
			int y2 = rectErrorPlot.bottom - (int)(rectErrorPlot.Height() * (m_plotData.error[i] - minError) / (maxError - minError));
			pDC->MoveTo(x1, y1);
			pDC->LineTo(x2, y2);
			// 如果超出显示范围，停止绘制
			if (m_plotData.time[i] >= displayTime)
				break;
		}
		// 实际角度曲线
		pDC->SelectObject(&penAngle);
		for (size_t i = 1; i < m_plotData.time.size(); i++)
		{
			// 只绘制displayTime范围内的数据
			if (m_plotData.time[i-1] > displayTime)
				break;
			int x1 = rectAnglePlot.left + (int)(rectAnglePlot.Width() * m_plotData.time[i-1] / displayTime);
			int y1 = rectAnglePlot.bottom - (int)(rectAnglePlot.Height() * (m_plotData.angle[i-1] - minAngle) / (maxAngle - minAngle));
			// 确保第二个点在显示范围内
			double time2 = min(m_plotData.time[i], displayTime);
			int x2 = rectAnglePlot.left + (int)(rectAnglePlot.Width() * time2 / displayTime);
			int y2 = rectAnglePlot.bottom - (int)(rectAnglePlot.Height() * (m_plotData.angle[i] - minAngle) / (maxAngle - minAngle));
			pDC->MoveTo(x1, y1);
			pDC->LineTo(x2, y2);
			// 如果超出显示范围，停止绘制
			if (m_plotData.time[i] >= displayTime)
				break;
		}
		// 参考角度曲线
		pDC->SelectObject(&penRefAngle);
		for (size_t i = 1; i < m_plotData.time.size(); i++)
		{
			// 只绘制displayTime范围内的数据
			if (m_plotData.time[i-1] > displayTime)
				break;
			int x1 = rectAnglePlot.left + (int)(rectAnglePlot.Width() * m_plotData.time[i-1] / displayTime);
			int y1 = rectAnglePlot.bottom - (int)(rectAnglePlot.Height() * (m_plotData.refAngle[i-1] - minAngle) / (maxAngle - minAngle));
			// 确保第二个点在显示范围内
			double time2 = min(m_plotData.time[i], displayTime);
			int x2 = rectAnglePlot.left + (int)(rectAnglePlot.Width() * time2 / displayTime);
			int y2 = rectAnglePlot.bottom - (int)(rectAnglePlot.Height() * (m_plotData.refAngle[i] - minAngle) / (maxAngle - minAngle));
			pDC->MoveTo(x1, y1);
			pDC->LineTo(x2, y2);
			// 如果超出显示范围，停止绘制
			if (m_plotData.time[i] >= displayTime)
				break;
		}
		// 添加图例
		pDC->SelectObject(&penAngle);
		pDC->MoveTo(rectAnglePlot.right - 100, rectAnglePlot.top - 20);
		pDC->LineTo(rectAnglePlot.right - 70, rectAnglePlot.top - 20);
		pDC->TextOut(rectAnglePlot.right - 65, rectAnglePlot.top - 25, _T("实际角度"));
		pDC->SelectObject(&penRefAngle);
		pDC->MoveTo(rectAnglePlot.right - 100, rectAnglePlot.top - 40);
		pDC->LineTo(rectAnglePlot.right - 70, rectAnglePlot.top - 40);
		pDC->TextOut(rectAnglePlot.right - 65, rectAnglePlot.top - 45, _T("目标角度"));
	}
	else
	{
		// 没有数据时，添加说明
		CFont promptFont;
		promptFont.CreateFont(m_nFontSize + 12, 0, 0, 0, FW_BOLD, FALSE, FALSE, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, m_strFontName);
		// 保存原字体并应用新字体
		CFont* pOldPromptFont = pDC->SelectObject(&promptFont);
		// 获取文本尺寸用于居中对齐
		CSize textSize = pDC->GetTextExtent(_T("点击开始按钮开始仿真"));
		// 计算居中位置
		int textX = rectTorquePlot.left + (rectTorquePlot.Width() - textSize.cx) / 2;
		int textY = rectTorquePlot.top + (rectTorquePlot.Height() - textSize.cy) / 2;
		// 显示提示文本
		pDC->TextOut(textX, textY, _T("点击开始按钮开始仿真"));
		// 恢复原字体
		pDC->SelectObject(pOldPromptFont);
		promptFont.DeleteObject();
	}
	// 恢复原画笔和字体
	pDC->SelectObject(pOldPen);
	pDC->SelectObject(pOldFont);
	// 释放GDI资源
	font.DeleteObject();
	penAxis.DeleteObject();
	penGrid.DeleteObject();
	penTorque.DeleteObject();
	penError.DeleteObject();
	penAngle.DeleteObject();
	penRefAngle.DeleteObject();
}

// 绘制膝关节动画
void CFinalView::DrawKneeAnimation(CDC* pDC, CRect& rectClient)
{
	// 计算动画区域(右半部分)
	CRect rectAnim = rectClient;
	rectAnim.left = rectClient.right / 2;
	// 创建字体
	CFont font;
	font.CreateFont(m_nFontSize + 4, 0, 0, 0, FW_BOLD, FALSE, FALSE, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, m_strFontName);
	// 保存原字体
	CFont* pOldFont = pDC->SelectObject(&font);
	// 绘制标题
	pDC->TextOut(rectAnim.left + 400, rectAnim.top + 200, _T("膝关节外骨骼控制动画"));
	// 绘制膝关节位置(画在中心偏上位置)
	int centerX = rectAnim.left + rectAnim.Width() / 2;
	int centerY = rectAnim.top + rectAnim.Height() / 3;
	// 创建画笔和画刷
	CPen penBody(PS_SOLID, 4, RGB(60, 60, 60));
	CPen penCalf(PS_SOLID, 4, RGB(0, 128, 0));
	CPen penTarget(PS_SOLID, 2, RGB(255, 0, 0));
	CPen penGround(PS_SOLID, 2, RGB(120, 100, 80));
	CBrush brushHead(RGB(255, 220, 180));
	CBrush brushJoint(RGB(180, 180, 220));
	// 保存原画笔和画刷
	CPen* pOldPen = pDC->SelectObject(&penBody);
	CBrush* pOldBrush = pDC->SelectObject(&brushHead);
	// 头部
	pDC->Ellipse(centerX - 20, centerY - 80, centerX + 20, centerY - 40);
	// 添加简单的面部特征
	pDC->SelectObject(GetStockObject(BLACK_PEN));
	// 眼睛
	pDC->Ellipse(centerX - 10, centerY - 70, centerX - 2, centerY - 62);
	pDC->Ellipse(centerX + 2, centerY - 70, centerX + 10, centerY - 62);
	// 嘴巴
	pDC->Arc(centerX - 8, centerY - 55, centerX + 8, centerY - 47, centerX - 8, centerY - 51, centerX + 8, centerY - 51);
	// 身体
	pDC->SelectObject(&penBody);
	pDC->MoveTo(centerX, centerY - 40);
	pDC->LineTo(centerX, centerY);
	// 绘制关节点
	pDC->SelectObject(&brushJoint);
	// 肩膀关节
	pDC->Ellipse(centerX - 6, centerY - 35, centerX + 6, centerY - 23);
	// 手臂
	pDC->SelectObject(&penBody);
	pDC->MoveTo(centerX, centerY - 30);
	pDC->LineTo(centerX - 30, centerY - 10);
	pDC->MoveTo(centerX, centerY - 30);
	pDC->LineTo(centerX + 30, centerY - 10);
	// 手
	pDC->SelectObject(&brushHead);
	pDC->Ellipse(centerX - 35, centerY - 15, centerX - 25, centerY - 5);
	pDC->Ellipse(centerX + 25, centerY - 15, centerX + 35, centerY - 5);
	// 计算大腿长度和小腿长度
	int thighLength = 80;
	int calfLength = 100;
	// 膝关节点
	pDC->SelectObject(&brushJoint);
	pDC->Ellipse(centerX - 6, centerY + thighLength - 6, centerX + 6, centerY + thighLength + 6);
	// 绘制大腿(固定垂直向下)
	pDC->SelectObject(&penBody);
	pDC->MoveTo(centerX, centerY);
	pDC->LineTo(centerX, centerY + thighLength);
	// 计算小腿位置
	double actualAngle = m_dCurrentAngle; // 实际角度(弧度)
	double targetAngle = GetTargetAngle(m_dCurrentTime); // 目标角度(弧度)
	// 小腿末端坐标(实际角度)
	int calfEndX = centerX + (int)(calfLength * sin(actualAngle));
	int calfEndY = centerY + thighLength + (int)(calfLength * cos(actualAngle));
	// 小腿末端坐标(目标角度)
	int targetEndX = centerX + (int)(calfLength * sin(targetAngle));
	int targetEndY = centerY + thighLength + (int)(calfLength * cos(targetAngle));
	// 绘制目标角度线
	pDC->SelectObject(&penTarget);
	pDC->MoveTo(centerX, centerY + thighLength);
	pDC->LineTo(targetEndX, targetEndY);
	// 绘制实际小腿
	pDC->SelectObject(&penCalf);
	pDC->MoveTo(centerX, centerY + thighLength);
	pDC->LineTo(calfEndX, calfEndY);
	// 绘制脚
	int footWidth = 30;
	int footHeight = 10;
	POINT footPoints[4] = {
		{calfEndX - 10, calfEndY},
		{calfEndX - 5, calfEndY + footHeight},
		{calfEndX + footWidth - 5, calfEndY + footHeight},
		{calfEndX + footWidth, calfEndY}
	};
	pDC->SelectObject(&brushHead);
	pDC->Polygon(footPoints, 4);
	// 绘制地面线
	pDC->SelectObject(&penGround);
	pDC->MoveTo(centerX - 100, calfEndY + footHeight + 5);
	pDC->LineTo(centerX + 150, calfEndY + footHeight + 5);
	// 绘制角度值
	CString strAngle;
	CFont fontInfo;
	fontInfo.CreateFont(m_nFontSize + 6, 0, 0, 0, FW_BOLD, FALSE, FALSE, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, m_strFontName);
	pDC->SelectObject(&fontInfo);
	strAngle.Format(_T("当前角度: %.1f°"), RAD2DEG(actualAngle));
	pDC->TextOut(rectAnim.left + 400, rectAnim.bottom - 440, strAngle);
	strAngle.Format(_T("目标角度: %.1f°"), RAD2DEG(targetAngle));
	pDC->TextOut(rectAnim.left + 400, rectAnim.bottom - 410, strAngle);
	strAngle.Format(_T("角度误差: %.1f°"), RAD2DEG(actualAngle - targetAngle));
	pDC->TextOut(rectAnim.left + 400, rectAnim.bottom - 380, strAngle);
	// 显示仿真时间
	CString strTime;
	strTime.Format(_T("仿真时间: %.2fs"), m_dCurrentTime);
	pDC->TextOut(rectAnim.left + 400, rectAnim.bottom - 350, strTime);
	// 恢复原画笔和字体
	pDC->SelectObject(pOldPen);
	pDC->SelectObject(pOldBrush);
	pDC->SelectObject(pOldFont);
	// 释放GDI资源
	font.DeleteObject();
	fontInfo.DeleteObject();
	penCalf.DeleteObject();
	penTarget.DeleteObject();
	penBody.DeleteObject();
	penGround.DeleteObject();
	brushHead.DeleteObject();
	brushJoint.DeleteObject();
}

// 获取目标角度
double CFinalView::GetTargetAngle(double time)
{
	if (m_bConstantAngle)
	{
		// 定值角度，直接返回设定的目标角度
		return m_dTargetAngle;
	}
	else
	{
		// 正弦角度，以目标角度为幅值生成正弦波
		return m_dTargetAngle * sin(2.0 * PI * time / m_dSinePeriod);
	}
}

// 获取控制扭矩(PID控制器)
double CFinalView::GetControlTorque(double currentAngle, double targetAngle)
{
	// 计算误差
	double error = targetAngle - currentAngle;
	// 积分误差
	m_dIntegralError += error * m_dTimeStep;
	// 微分误差
	double derivativeError = (error - m_dLastError) / m_dTimeStep;
	m_dLastError = error;
	// PID控制律
	double torque = m_dKp * error + m_dKi * m_dIntegralError + m_dKd * derivativeError;
	return torque;
}

// 系统动力学方程
void CFinalView::SystemDynamics(double theta, double omega, double tau, double& dtheta, double& domega)
{
	// theta: 角度
	// tau: 输入扭矩
	// dtheta: 角度导数
	// domega: 角速度导数 = omega: 角速度
	dtheta = omega;
	// 角速度导数 = (输入扭矩 - 重力扭矩*sin(角度) - 固定摩擦系数*sgn(角速度) - 阻尼系数*角速度) / 惯性矩
	double frictionTerm = (omega > 0) ? m_dA : ((omega < 0) ? -m_dA : 0);
	domega = (tau - m_dTauG * sin(theta) - frictionTerm - m_dB * omega) / m_dJ;
}

// 欧拉法更新系统状态
void CFinalView::EulerUpdate()
{
	// 获取目标角度
	double targetAngle = GetTargetAngle(m_dCurrentTime);
	// 计算控制扭矩
	double torque = GetControlTorque(m_dCurrentAngle, targetAngle);
	// 计算导数
	double dtheta, domega;
	SystemDynamics(m_dCurrentAngle, m_dCurrentVelocity, torque, dtheta, domega);
	// 欧拉法更新状态
	m_dCurrentAngle += dtheta * m_dTimeStep;
	m_dCurrentVelocity += domega * m_dTimeStep;
	// 记录数据点
	m_plotData.time.push_back(m_dCurrentTime);
	m_plotData.torque.push_back(torque);
	m_plotData.error.push_back(targetAngle - m_dCurrentAngle);
	m_plotData.angle.push_back(m_dCurrentAngle);
	m_plotData.refAngle.push_back(targetAngle);
	// 更新时间
	m_dCurrentTime += m_dTimeStep;
}

// 四阶龙格库塔法更新系统状态
void CFinalView::RK4Update()
{
	// 获取目标角度
	double targetAngle = GetTargetAngle(m_dCurrentTime);
	// 计算控制扭矩
	double torque = GetControlTorque(m_dCurrentAngle, targetAngle);
	// 保存初始状态
	double theta = m_dCurrentAngle;
	double omega = m_dCurrentVelocity;
	// 计算k1
	double k1_theta, k1_omega;
	SystemDynamics(theta, omega, torque, k1_theta, k1_omega);
	// 计算k2
	double k2_theta, k2_omega;
	SystemDynamics(theta + 0.5 * m_dTimeStep * k1_theta, omega + 0.5 * m_dTimeStep * k1_omega, torque, k2_theta, k2_omega);
	// 计算k3
	double k3_theta, k3_omega;
	SystemDynamics(theta + 0.5 * m_dTimeStep * k2_theta, omega + 0.5 * m_dTimeStep * k2_omega, torque, k3_theta, k3_omega);
	// 计算k4
	double k4_theta, k4_omega;
	SystemDynamics(theta + m_dTimeStep * k3_theta, omega + m_dTimeStep * k3_omega, torque, k4_theta, k4_omega);
	// 更新状态
	m_dCurrentAngle = theta + m_dTimeStep * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta) / 6.0;
	m_dCurrentVelocity = omega + m_dTimeStep * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega) / 6.0;
	// 记录数据点
	m_plotData.time.push_back(m_dCurrentTime);
	m_plotData.torque.push_back(torque);
	m_plotData.error.push_back(targetAngle - m_dCurrentAngle);
	m_plotData.angle.push_back(m_dCurrentAngle);
	m_plotData.refAngle.push_back(targetAngle);
	// 更新时间
	m_dCurrentTime += m_dTimeStep;
}

// 菜单事件处理程序

void CFinalView::OnMenuBasic()
{
	// 创建基本设置对话框
	CBasicDlg dlg;
	// 设置对话框初始值
	dlg.m_nFontSize = m_nFontSize;
	dlg.m_nXInterval = m_nXInterval;
	dlg.m_nAxisWidth = m_nAxisWidth;
	dlg.m_colorAxis = m_colorAxis;
	dlg.m_strFontName = m_strFontName;
	// 显示对话框
	if (dlg.DoModal() == IDOK)
	{
		// 获取用户设置
		m_nFontSize = dlg.m_nFontSize;
		m_nXInterval = dlg.m_nXInterval;
		m_nAxisWidth = dlg.m_nAxisWidth;
		m_colorAxis = dlg.m_colorAxis;
		m_strFontName = dlg.m_strFontName;
		// 刷新视图
		Invalidate();
	}
}

void CFinalView::OnMenuParameter()
{
	// 创建参数设置对话框
	CParameterDlg dlg;
	// 设置对话框初始值
	dlg.m_dJ = m_dJ;
	dlg.m_dB = m_dB;
	dlg.m_dA = m_dA;
	dlg.m_dTauG = m_dTauG;
	dlg.m_dInitialAngle = RAD2DEG(m_dInitialAngle);
	// 显示对话框
	if (dlg.DoModal() == IDOK)
	{
		// 获取用户设置
		m_dJ = dlg.m_dJ;
		m_dB = dlg.m_dB;
		m_dA = dlg.m_dA;
		m_dTauG = dlg.m_dTauG;
		m_dInitialAngle = DEG2RAD(dlg.m_dInitialAngle);
		// 更新当前角度为新设置的初始角度
		if (!m_bSimulating)
		{
			m_dCurrentAngle = m_dInitialAngle;
		}
		// 立即刷新视图以显示新的参数值
		Invalidate(TRUE);
		UpdateWindow();
	}
}

void CFinalView::OnMenuSimulation()
{
	// 创建仿真设置对话框
	CSimulationDlg dlg;
	// 设置对话框初始值
	dlg.m_dSimTime = m_dSimTime;
	dlg.m_dTimeStep = m_dTimeStep;
	dlg.m_bConstantAngle = !m_bConstantAngle;
	dlg.m_dTargetAngle = RAD2DEG(m_dTargetAngle);
	dlg.m_dSinePeriod = m_dSinePeriod;
	dlg.m_nSimMethod = m_bRK4 ? 1 : 0;
	dlg.m_dKp = m_dKp;
	dlg.m_dKi = m_dKi;
	dlg.m_dKd = m_dKd;
	// 复制PID参数记录到对话框
	dlg.m_arrPIDParams.RemoveAll();
	for (int i = 0; i < m_arrPIDParams.GetSize(); i++)
	{
		dlg.m_arrPIDParams.Add(m_arrPIDParams[i]);
	}
	// 显示对话框
	if (dlg.DoModal() == IDOK)
	{
		// 获取用户设置
		m_dSimTime = dlg.m_dSimTime;
		m_dTimeStep = dlg.m_dTimeStep;
		m_bConstantAngle = !(dlg.m_bConstantAngle == TRUE);
		m_dTargetAngle = DEG2RAD(dlg.m_dTargetAngle);
		m_dSinePeriod = dlg.m_dSinePeriod;
		m_bRK4 = dlg.m_nSimMethod == 1;
		// 获取PID参数
		if (dlg.m_nSelectedPID >= 0 && dlg.m_nSelectedPID < dlg.m_arrPIDParams.GetSize())
		{
			// 如果有选中的PID参数，使用选中的参数
			m_dKp = dlg.m_arrPIDParams[dlg.m_nSelectedPID].kp;
			m_dKi = dlg.m_arrPIDParams[dlg.m_nSelectedPID].ki;
			m_dKd = dlg.m_arrPIDParams[dlg.m_nSelectedPID].kd;
		}
		else
		{
			// 否则使用输入框中的值
			m_dKp = dlg.m_dKp;
			m_dKi = dlg.m_dKi;
			m_dKd = dlg.m_dKd;
		}
		// 保存对话框中的PID参数记录
		m_arrPIDParams.RemoveAll();
		for (int i = 0; i < dlg.m_arrPIDParams.GetSize(); i++)
		{
			m_arrPIDParams.Add(dlg.m_arrPIDParams[i]);
		}
		
		// 立即刷新视图以显示新的参数值
		Invalidate(TRUE);
		UpdateWindow();
	}
}

// 保存设置
void CFinalView::SaveSettings(LPCTSTR lpszFileName)
{
	try
	{
		// 打开文件
		CFile file(lpszFileName, CFile::modeCreate | CFile::modeWrite);
		CArchive ar(&file, CArchive::store);
		// 系统参数
		ar << m_dJ << m_dB << m_dA << m_dTauG << m_dInitialAngle;
		// 仿真参数
		ar << m_dSimTime << m_dTimeStep << (BOOL)m_bConstantAngle << m_dTargetAngle << m_dSinePeriod << (BOOL)m_bRK4;
		// PID参数
		ar << m_dKp << m_dKi << m_dKd;
		// PID参数记录
		int nCount = m_arrPIDParams.GetSize();
		ar << nCount;
		for (int i = 0; i < nCount; i++)
		{
			ar << m_arrPIDParams[i].kp << m_arrPIDParams[i].ki 
			   << m_arrPIDParams[i].kd << m_arrPIDParams[i].eval;
		}
		// 图形设置
		ar << m_strFontName << m_nFontSize << m_nXInterval << m_nAxisWidth << m_colorAxis;
		// 关闭文件
		ar.Close();
		file.Close();
		// 提示保存成功
		MessageBox(_T("设置已保存"), _T("保存成功"), MB_ICONINFORMATION | MB_OK);
	}
	catch (CFileException* e)
	{
		// 处理文件异常
		TCHAR szError[256];
		e->GetErrorMessage(szError, 256);
		MessageBox(szError, _T("保存失败"), MB_ICONERROR | MB_OK);
		e->Delete();
	}
}

void CFinalView::OnMenuSave()
{
	// 创建文件对话框
	CFileDialog dlg(FALSE, _T("dat"), _T("KneeSimulation.dat"),
		OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT,
		_T("Data Files (*.dat)|*.dat|All Files (*.*)|*.*||"));
	// 显示对话框
	if (dlg.DoModal() == IDOK)
	{
		// 保存设置
		SaveSettings(dlg.GetPathName());
	}
}

void CFinalView::OnMenuLoad()
{
	// 创建文件对话框
	CFileDialog dlg(TRUE, _T("dat"), _T("KneeSimulation.dat"),
		OFN_HIDEREADONLY | OFN_FILEMUSTEXIST,
		_T("Data Files (*.dat)|*.dat|All Files (*.*)|*.*||"));
	// 显示对话框
	if (dlg.DoModal() == IDOK)
	{
		// 加载设置
		LoadSettings(dlg.GetPathName());
		// 刷新视图
		Invalidate();
	}
}

// 加载设置
void CFinalView::LoadSettings(LPCTSTR lpszFileName)
{
	try
	{
		// 打开文件
		CFile file(lpszFileName, CFile::modeRead);
		CArchive ar(&file, CArchive::load);
		// 系统参数
		ar >> m_dJ >> m_dB >> m_dA >> m_dTauG >> m_dInitialAngle;
		// 仿真参数
		BOOL bConstant, bRK4;
		ar >> m_dSimTime >> m_dTimeStep >> bConstant
		   >> m_dTargetAngle >> m_dSinePeriod >> bRK4;
		m_bConstantAngle = bConstant != FALSE;
		m_bRK4 = bRK4 != FALSE;
		// PID参数
		ar >> m_dKp >> m_dKi >> m_dKd;
		// PID参数记录
		int nCount;
		ar >> nCount;
		m_arrPIDParams.RemoveAll();
		for (int i = 0; i < nCount; i++)
		{
			PIDParams params;
			ar >> params.kp >> params.ki >> params.kd >> params.eval;
			m_arrPIDParams.Add(params);
		}
		// 图形设置
		ar >> m_strFontName >> m_nFontSize >> m_nXInterval >> m_nAxisWidth >> m_colorAxis;
		// 关闭文件
		ar.Close();
		file.Close();
		// 提示加载成功
		MessageBox(_T("设置已加载"), _T("加载成功"), MB_ICONINFORMATION | MB_OK);
	}
	catch (CFileException* e)
	{
		// 处理文件异常
		TCHAR szError[256];
		e->GetErrorMessage(szError, 256);
		MessageBox(szError, _T("加载失败"), MB_ICONERROR | MB_OK);
		e->Delete();
	}
}

// 工具栏按钮事件处理程序
void CFinalView::OnButtonStart()
{
	// 如果已经在仿真，先停止
	if (m_bSimulating)
	{
		StopSimulation();
	}
	// 初始化并开始仿真
	InitSimulation();
}

void CFinalView::OnButtonPause()
{
	// 仅在仿真中才能暂停/继续
	if (m_bSimulating)
	{
		if (m_bPaused)
		{
			// 继续仿真
			m_bPaused = false;
		}
		else
		{
			// 暂停仿真
			m_bPaused = true;
		}
	}
}

void CFinalView::OnButtonStop()
{
	if (m_bSimulating || m_bSimulationCompleted)
	{
		m_bSimulationCompleted = false; 
		StopSimulation();
		// 强制立即刷新视图
		Invalidate(TRUE);
		UpdateWindow();
	}
}

// 定时器事件处理程序
void CFinalView::OnTimer(UINT_PTR nIDEvent)
{
	if (nIDEvent == ID_TIMER_SIMULATION)
	{
		// 如果暂停或已经结束，不更新
		if (m_bPaused || !m_bSimulating || m_dCurrentTime >= m_dSimTime)
		{
			// 如果超过仿真时间，停止仿真
			if (m_dCurrentTime >= m_dSimTime)
			{
				m_bSimulationCompleted = true;
				StopSimulation();
			}
			return;
		}
		// 执行多个仿真步骤，减少刷新频率，避免闪烁
		int steps = 10; 
		for (int i = 0; i < steps; i++)
		{
			// 根据选择的方法更新系统状态
			if (m_bRK4)
			{
				RK4Update();
			}
			else
			{
				EulerUpdate();
			}
			// 如果仿真时间已经结束，跳出循环
			if (m_dCurrentTime >= m_dSimTime)
			{
				m_bSimulationCompleted = true; 
				StopSimulation();
				break;
			}
		}
		Invalidate(FALSE);
	}
	CView::OnTimer(nIDEvent);
}

// CFinalView 打印

BOOL CFinalView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 默认准备
	return DoPreparePrinting(pInfo);
}

void CFinalView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 添加额外的打印前进行的初始化过程
}

void CFinalView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 添加打印后进行的清理过程
}

// CFinalView 诊断

#ifdef _DEBUG
void CFinalView::AssertValid() const
{
	CView::AssertValid();
}

void CFinalView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CFinalDoc* CFinalView::GetDocument() const // 非调试版本是内联的
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CFinalDoc)));
	return (CFinalDoc*)m_pDocument;
}
#endif //_DEBUG


// CFinalView 消息处理程序
// 
// 创建双缓冲
void CFinalView::CreateBuffer(CDC* pDC)
{
	// 获取客户区大小
	GetClientRect(&m_rectClient);
	// 如果已经创建过缓冲区，先删除
	if (m_bBufferCreated)
	{
		DeleteBuffer();
	}
	// 创建兼容DC
	m_memDC.CreateCompatibleDC(pDC);
	// 创建兼容位图
	m_memBitmap.CreateCompatibleBitmap(pDC, m_rectClient.Width(), m_rectClient.Height());
	// 选择位图到内存DC
	m_pOldBitmap = m_memDC.SelectObject(&m_memBitmap);
	// 标记缓冲区已创建
	m_bBufferCreated = true;
	// 设置白色背景
	m_memDC.FillSolidRect(m_rectClient, RGB(255, 255, 255));
}

// 销毁双缓冲
void CFinalView::DeleteBuffer()
{
	if (m_bBufferCreated)
	{
		// 恢复原位图
		if (m_pOldBitmap != NULL)
		{
			m_memDC.SelectObject(m_pOldBitmap);
			m_pOldBitmap = NULL;
		}
		// 删除位图和DC
		m_memBitmap.DeleteObject();
		m_memDC.DeleteDC();
		// 标记缓冲区已删除
		m_bBufferCreated = false;
	}
}

// 处理窗口大小变化
void CFinalView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);
	CDC* pDC = GetDC();
	if (pDC != NULL)
	{
		DeleteBuffer();
		CreateBuffer(pDC);
	    ReleaseDC(pDC);
	}
}

// CFinalView 初始化更新
void CFinalView::OnInitialUpdate()
{
	CView::OnInitialUpdate();
	// 通过消息延迟调用基本设置对话框，确保主窗口已显示
	PostMessage(WM_SHOW_BASIC);
}
 

// 初始调用基本设置对话框
LRESULT CFinalView::OnShowBasic(WPARAM wParam, LPARAM lParam)
{
	OnMenuBasic();
	return 0;
}

// 添加新的 DrawParameterInfo 函数用于绘制参数信息
void CFinalView::DrawParameterInfo(CDC* pDC, CRect& rectClient)
{
	// 计算参数信息区域（位于左右两侧中间）
	CRect rectParam = rectClient;
	int midPoint = rectClient.right / 2;
	int panelWidth = 240;  // 更合适的宽度
	
	rectParam.left = midPoint;  // 居中显示
	rectParam.right = midPoint + panelWidth;
	rectParam.top = rectClient.top + 200;     
	rectParam.bottom = rectClient.top + 650;   // 适当的高度
	
	// 创建字体
	CFont fontTitle, fontContent, fontGroup;
	fontTitle.CreateFont(m_nFontSize + 6, 0, 0, 0, FW_BOLD, FALSE, FALSE, 0,
		DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
		DEFAULT_PITCH | FF_DONTCARE, m_strFontName);
	
	fontContent.CreateFont(m_nFontSize + 2, 0, 0, 0, FW_NORMAL, FALSE, FALSE, 0,
		DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
		DEFAULT_PITCH | FF_DONTCARE, m_strFontName);

	fontGroup.CreateFont(m_nFontSize + 4, 0, 0, 0, FW_BOLD, FALSE, TRUE, 0,
		DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
		DEFAULT_PITCH | FF_DONTCARE, m_strFontName);
	
	// 创建画笔和画刷
	CPen penBorder(PS_SOLID, 2, RGB(100, 100, 100));
	CPen penSystem(PS_SOLID, 3, RGB(0, 120, 0));      // 加粗绿色表示系统参数
	CPen penSimulation(PS_SOLID, 3, RGB(0, 0, 200));  // 加粗蓝色表示仿真参数
	CPen penPID(PS_SOLID, 3, RGB(200, 0, 0));         // 加粗红色表示PID参数
	CBrush brushBg(RGB(252, 252, 245));               // 更淡的背景颜色
	
	// 保存原画笔和画刷
	CPen* pOldPen = pDC->SelectObject(&penBorder);
	CBrush* pOldBrush = pDC->SelectObject(&brushBg);
	CFont* pOldFont = pDC->SelectObject(&fontTitle);
	
	// 设置参数面板背景，使用圆角矩形
	pDC->RoundRect(rectParam, CPoint(15, 15));
	
	// 设置文本背景模式为透明
	int oldBkMode = pDC->SetBkMode(TRANSPARENT);
	
	// 绘制标题
	CString strTitle = _T("当前参数设置");
	CSize sizeTitle = pDC->GetTextExtent(strTitle);
	pDC->TextOut(rectParam.left + (rectParam.Width() - sizeTitle.cx) / 2, rectParam.top + 10, strTitle);
	
	// 计算内容区域
	CRect rectContent = rectParam;
	rectContent.DeflateRect(10, sizeTitle.cy + 20, 10, 10);
	
	// 设置分组间隔
	int groupSpacing = 20;
	int lineHeight = m_nFontSize + 6;
	int yPos = rectContent.top;
	CString strValue;

	// 绘制系统参数分组标题
	pDC->SelectObject(&penSystem);
	pDC->MoveTo(rectContent.left, yPos + lineHeight / 2);
	
	pDC->SelectObject(&fontGroup);
	pDC->SetTextColor(RGB(0, 120, 0));
	pDC->TextOut(rectContent.left, yPos, _T("系统参数"));
	yPos += lineHeight + 5;
	
	// 绘制系统参数
	pDC->SelectObject(&fontContent);
	pDC->SetTextColor(RGB(0, 0, 0));
	
	strValue.Format(_T("惯性矩(J): %.2f kg·m²"), m_dJ);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight;
	
	strValue.Format(_T("阻尼系数(B): %.2f N·m·s/rad"), m_dB);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight;
	
	strValue.Format(_T("摩擦系数(A): %.2f N·m"), m_dA);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight;
	
	strValue.Format(_T("重力扭矩(τG): %.2f N·m"), m_dTauG);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight;
	
	strValue.Format(_T("初始角度: %.1f°"), RAD2DEG(m_dInitialAngle));
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight + groupSpacing;
	
	// 绘制仿真参数分组标题
	pDC->SelectObject(&penSimulation);
	pDC->MoveTo(rectContent.left, yPos + lineHeight / 2);
	
	pDC->SelectObject(&fontGroup);
	pDC->SetTextColor(RGB(0, 0, 200));
	pDC->TextOut(rectContent.left, yPos, _T("仿真参数"));
	yPos += lineHeight + 5;
	
	// 绘制仿真参数
	pDC->SelectObject(&fontContent);
	pDC->SetTextColor(RGB(0, 0, 0));
	
	strValue.Format(_T("仿真时间: %.1f s"), m_dSimTime);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight;
	
	strValue.Format(_T("仿真步长: %.3f s"), m_dTimeStep);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight;
	
	CString strAngleType = m_bConstantAngle ? _T("定值") : _T("正弦");
	strValue.Format(_T("目标角度类型: %s"), strAngleType);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight;
	
	if (!m_bConstantAngle) {
		strValue.Format(_T("正弦周期: %.1f s"), m_dSinePeriod);
		pDC->TextOut(rectContent.left + 10, yPos, strValue);
		yPos += lineHeight;
	}
	
	CString strMethod = m_bRK4 ? _T("四阶龙格库塔法") : _T("欧拉法");
	strValue.Format(_T("积分方法: %s"), strMethod);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight + groupSpacing;
	
	// 绘制PID参数分组标题
	pDC->SelectObject(&penPID);
	pDC->MoveTo(rectContent.left, yPos + lineHeight / 2);
	
	pDC->SelectObject(&fontGroup);
	pDC->SetTextColor(RGB(200, 0, 0));
	pDC->TextOut(rectContent.left, yPos, _T("PID控制参数"));
	yPos += lineHeight + 5;
	
	// 绘制PID参数
	pDC->SelectObject(&fontContent);
	pDC->SetTextColor(RGB(0, 0, 0));
	
	strValue.Format(_T("比例系数(Kp): %.2f"), m_dKp);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight;
	
	strValue.Format(_T("积分系数(Ki): %.2f"), m_dKi);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight;
	
	strValue.Format(_T("微分系数(Kd): %.2f"), m_dKd);
	pDC->TextOut(rectContent.left + 10, yPos, strValue);
	yPos += lineHeight + 10;
	
	// 恢复原始设置
	pDC->SetBkMode(oldBkMode);
	pDC->SelectObject(pOldPen);
	pDC->SelectObject(pOldBrush);
	pDC->SelectObject(pOldFont);
	
	// 释放GDI资源
	fontTitle.DeleteObject();
	fontContent.DeleteObject();
	fontGroup.DeleteObject();
	penBorder.DeleteObject();
	penSystem.DeleteObject();
	penSimulation.DeleteObject();
	penPID.DeleteObject();
	brushBg.DeleteObject();
}