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

static void usage(const char* program) {
    std::cout << "usage: " << program << " [photos-folder] [--target DIR] [--debug DIR]\n"
              << "\n"
              << "  photos-folder, --target DIR  photos (jpg/png) of the person to follow. Only the\n"
              << "                               person whose face matches them is followed. Defaults\n"
              << "                               to ./res/target when that folder exists. Without any\n"
              << "                               photos the first person seen is followed and their\n"
              << "                               face and clothes are learned from the camera\n"
              << "  --debug DIR                  write an annotated camera frame per second to DIR\n"
              << "                               (tracks, faces, clothing regions, similarities)\n"
              << "  --no-target                  ignore ./res/target, follow the first person seen\n"
              << std::endl;
}

int main(int argc, char** argv)
{
    std::string targetImages;
    std::string debugDir;
    bool noTarget = false;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--target" && i + 1 < argc) {
            targetImages = argv[++i];
        } else if (arg == "--debug" && i + 1 < argc) {
            debugDir = argv[++i];
        } else if (arg == "--no-target") {
            noTarget = true;
        } else if (!arg.empty() && arg[0] != '-' && targetImages.empty()) {
            targetImages = arg;
        } else {
            std::cerr << "unknown argument " << arg << std::endl;
            usage(argv[0]);
            return 1;
        }
    }
    // photos of the person to follow: the folder given as argument, else ./res/target if
    // it exists. Without any, the first person that shows up is followed
    if (targetImages.empty() && !noTarget && std::filesystem::is_directory("./res/target"))
        targetImages = "./res/target";
    if (!targetImages.empty() && !std::filesystem::is_directory(targetImages)) {
        std::cerr << "photos folder " << targetImages << " does not exist" << std::endl;
        return 1;
    }

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

    examples::FollowPerson trick(control, targetImages, debugDir);
    trick.run();

    //examples::SitStandApproach trick(control);
    //trick.run();

    return 0;
}
