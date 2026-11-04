#include <iostream>

#include "core/Control.h"
#include "tricks/TurnToPerson.h"
#include "tricks/SitStandApproach.h"

using namespace robo::sdk;
using namespace robo::sdk::core;

int main(int argc, char** argv)
{
    std::cout << "Start" << std::endl;

    unitree::robot::ChannelFactory::Instance()->Init(0, "eth0");

    Control control;
    control.init();
    sleep(1);

    control.printStatus();
    if (!control.standup()) {
        std::cout << "error in standup" << std::endl;
    }
    control.printStatus();

    examples::TurnToPerson trick(control);
    trick.run();

    //examples::SitStandApproach trick(control);
    //trick.run();

    return 0;
}
