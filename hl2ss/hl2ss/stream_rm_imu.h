
#pragma once

#include <researchmode/ResearchModeApi.h>
#include <WinSock2.h>

#include <winrt/Windows.Perception.Spatial.h>

void RM_ACC_Mode0(IResearchModeSensor* sensor, SOCKET clientsocket);
void RM_ACC_Mode1(IResearchModeSensor* sensor, SOCKET clientsocket, winrt::Windows::Perception::Spatial::SpatialLocator const& locator);
void RM_ACC_Mode2(IResearchModeSensor* sensor, SOCKET clientsocket);

void RM_GYR_Mode0(IResearchModeSensor* sensor, SOCKET clientsocket);
void RM_GYR_Mode1(IResearchModeSensor* sensor, SOCKET clientsocket, winrt::Windows::Perception::Spatial::SpatialLocator const& locator);
void RM_GYR_Mode2(IResearchModeSensor* sensor, SOCKET clientsocket);

void RM_MAG_Mode0(IResearchModeSensor* sensor, SOCKET clientsocket);
void RM_MAG_Mode1(IResearchModeSensor* sensor, SOCKET clientsocket, winrt::Windows::Perception::Spatial::SpatialLocator const& locator);
void RM_MAG_Mode2(IResearchModeSensor* sensor, SOCKET clientsocket);
