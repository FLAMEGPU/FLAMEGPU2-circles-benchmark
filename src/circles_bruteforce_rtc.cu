#include <algorithm>

#include "flamegpu/flame_api.h"
#include "common.cuh"

namespace {
const char * output_message = R"###(
FLAMEGPU_AGENT_FUNCTION(output_message, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    return ALIVE;
}
)###";

const char * move = R"###(
FLAMEGPU_AGENT_FUNCTION(move, MsgBruteForce, MsgNone) {
    const int ID = FLAMEGPU->getVariable<int>("id");
    const float REPULSE_FACTOR = FLAMEGPU->environment.getProperty<float>("repulse");
    const float RADIUS = FLAMEGPU->environment.getProperty<float>("radius");
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");
    int count = 0;
    for (const auto &message : FLAMEGPU->message_in) {
        if (message.getVariable<int>("id") != ID) {
            const float x2 = message.getVariable<float>("x");
            const float y2 = message.getVariable<float>("y");
            const float z2 = message.getVariable<float>("z");
            float x21 = x2 - x1;
            float y21 = y2 - y1;
            float z21 = z2 - z1;
            const float separation = cbrt(x21*x21 + y21*y21 + z21*z21);
            if (separation < RADIUS && separation > 0.0f) {
                float k = sinf((separation / RADIUS)*3.141*-2)*REPULSE_FACTOR;
                // Normalise without recalculating separation
                x21 /= separation;
                y21 /= separation;
                z21 /= separation;
                fx += k * x21;
                fy += k * y21;
                fz += k * z21;
                count++;
            }
        }
    }
    fx /= count > 0 ? count : 1;
    fy /= count > 0 ? count : 1;
    fz /= count > 0 ? count : 1;
    FLAMEGPU->setVariable<float>("x", x1 + fx);
    FLAMEGPU->setVariable<float>("y", y1 + fy);
    FLAMEGPU->setVariable<float>("z", z1 + fz);
    FLAMEGPU->setVariable<float>("drift", cbrt(fx*fx + fy*fy + fz*fz));
    return ALIVE;
}
)###";

#if defined(CIRCLES_VALIDATION) && CIRCLES_VALIDATION
FLAMEGPU_STEP_FUNCTION(Validation) {
    static float prevTotalDrift = FLT_MAX;
    static unsigned int driftDropped = 0;
    static unsigned int driftIncreased = 0;
    // This value should decline? as the model moves towards a steady equlibrium state
    // Once an equilibrium state is reached, it is likely to oscillate between 2-4? values
    float totalDrift = FLAMEGPU->agent("Circle").sum<float>("drift");
    if (totalDrift <= prevTotalDrift)
        driftDropped++;
    else
        driftIncreased++;
    prevTotalDrift = totalDrift;
    // printf("Avg Drift: %g\n", totalDrift / FLAMEGPU->agent("Circle").count());
    printf("%.2f%% Drift correct\n", 100 * driftDropped / static_cast<float>(driftDropped + driftIncreased));
}
#endif 
}  // namespace

// Run an individual simulation, using 
void run_circles_bruteforce_rtc(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs){
    ModelDescription model("circles_bruteforce_rtc");
    // Calculate environment bounds.
    const float ENV_WIDTH = runInputs.ENV_WIDTH;
    const float ENV_MIN = -0.5 * ENV_WIDTH;
    const float ENV_MAX = ENV_MIN + ENV_WIDTH;
    // Compute the actual density and return it.
    runOutputs.agentDensity = runInputs.AGENT_COUNT / (ENV_WIDTH * ENV_WIDTH * ENV_WIDTH);

    {   // Location message
        MsgBruteForce::Description &message = model.newMessage<MsgBruteForce>("location");
        message.newVariable<int>("id");
        message.newVariable<float>("x");
        message.newVariable<float>("y");
        message.newVariable<float>("z");
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("Circle");
        agent.newVariable<int>("id");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("z");
        agent.newVariable<float>("drift");  // Store the distance moved here, for validation
        agent.newRTCFunction("output_message", output_message).setMessageOutput("location");
        agent.newRTCFunction("move", move).setMessageInput("location");
    }

    // Global environment variables.
    {
        EnvironmentDescription &env = model.Environment();
        env.newProperty("repulse", ENV_REPULSE);
        env.newProperty("radius", runInputs.COMM_RADIUS);
    }

    // Organise the model. 

#if defined(CIRCLES_VALIDATION) && CIRCLES_VALIDATION
    {   // Attach init/step/exit functions and exit condition
        model.addStepFunction(Validation);
    }
#endif  // CIRCLES_VALIDATION

    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction("Circle", "output_message");
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction("Circle", "move");
    }

    // Create the simulation object
    CUDASimulation simulation(model);

    // Set config configuraiton properties 
    simulation.SimulationConfig().timing = false;
    simulation.SimulationConfig().verbose = false;
    simulation.SimulationConfig().random_seed = runInputs.HOST_SEED;  // @todo device seed != host seed? 
    simulation.SimulationConfig().steps = runInputs.STEPS;
    simulation.CUDAConfig().device_id = runInputs.CUDA_DEVICE;

    // Generate the initial population
    std::default_random_engine rng(runInputs.HOST_SEED);
    std::uniform_real_distribution<float> dist(ENV_MIN, ENV_MAX);
    AgentVector population(model.Agent("Circle"), runInputs.AGENT_COUNT);
    for (unsigned int i = 0; i < runInputs.AGENT_COUNT; i++) {
        AgentVector::Agent instance = population[i];
        instance.setVariable<int>("id", i);
        instance.setVariable<float>("x", dist(rng));
        instance.setVariable<float>("y", dist(rng));
        instance.setVariable<float>("z", dist(rng));
    }

    // Set the population for the simulation.
    simulation.setPopulationData(population);

    // Execute 
    simulation.simulate();

    // Store timing information for later use.
    runOutputs.ms_rtc = simulation.getElapsedTimeRTCInitialisation();
    runOutputs.ms_simulation = simulation.getElapsedTimeSimulation();
    runOutputs.ms_init = simulation.getElapsedTimeInitFunctions();
    runOutputs.ms_exit = simulation.getElapsedTimeExitFunctions();
    
    std::vector<float> ms_steps = simulation.getElapsedTimeSteps();
    runOutputs.ms_per_step = std::make_shared<std::vector<float>>(std::vector<float>(ms_steps.begin(), ms_steps.end()));
    runOutputs.ms_stepMean = std::accumulate(ms_steps.begin(), ms_steps.end(), 0.f) / (float)simulation.getStepCounter();
    runOutputs.mean_messageCount = runInputs.AGENT_COUNT;
}