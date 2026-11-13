#include <csignal>
#include <filesystem>
#include <iostream>
#include <string>

#include "core/Control.h"
#include "tricks/TurnToPerson.h"
#include "tricks/SitStandApproach.h"
#include "tricks/FollowPerson.h"

using namespace robo::sdk;
using namespace robo::sdk::core;

int main(int argc, char** argv)
{
    std::cout << "Start" << std::endl;

    // without this Ctrl+C would leave the avoidance service ignoring the remote controller
    std::signal(SIGINT, [](int) { Control::stopRequested = true; });
    std::signal(SIGTERM, [](int) { Control::stopRequested = true; });

    unitree::robot::ChannelFactory::Instance()->Init(0, "eth0");

    Control control;
    control.init();
    sleep(1);

    control.printStatus();
    if (!control.standup()) {
        std::cout << "error in standup" << std::endl;
    }
    control.printStatus();

    // images of the person to follow: the folder given as argument, else ./res/target if
    // it exists. Without any, the first person that shows up is followed
    std::string targetImages = argc > 1 ? argv[1] : "";
    if (targetImages.empty() && std::filesystem::is_directory("./res/target"))
        targetImages = "./res/target";

    examples::FollowPerson trick(control, targetImages);
    trick.run();

    //examples::SitStandApproach trick(control);
    //trick.run();

    return 0;
}
