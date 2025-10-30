#pragma once
#include "dataBase.h"

class solverBase :
    public dataBase
{
public:
    solverBase() : dataBase()
	{
	}

	~solverBase()
	{
	}

    void solve();

protected:
	void setProblemName(const std::string& name)
	{
		dir = name + "Output";
	}

    void outputFluidVTU();

	void outputSolidVTU();

    void outputFluid2SolidVTU();

    virtual void conditionInitialize()
    {
    }

    virtual void outputData()
    {
    }

    virtual void handleDataAfterCalculateContact()
    {
    }

private:
	std::string dir = "problem";

    void initialize();

    virtual void update();
};