#pragma once
#include "data.h"

class solverBase :
    public data
{
public:
    solverBase() : data()
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
