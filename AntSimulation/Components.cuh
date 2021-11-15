///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: Components.cuh
// Description: The header file for all of the simulation componets.
// 
// Change Log:
//	- 15/11/2021:JS - Added in block comments.
//
// Online sources:  
//	- (URL)
// 
// 
//////////////////////////// 80 columns wide //////////////////////////////////
#pragma once
#define LEAVING_HOME 0;
#define FOUND_FOOD 1;

struct MoveComponent
{
    float x, y;
    float speed, rotation;
};

struct SniffComponent
{
    float sniffMaxDistance;
};

struct ActivityComponent
{

};
