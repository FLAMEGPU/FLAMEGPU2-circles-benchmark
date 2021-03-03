#include <algorithm>

#include "flamegpu/flame_api.h"
#include "common.cuh"

namespace {

FLAMEGPU_AGENT_FUNCTION(output_message, MsgNone, MsgSpatial3D) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setLocation(
    FLAMEGPU->getVariable<float>("x"),
    FLAMEGPU->getVariable<float>("y"),
    FLAMEGPU->getVariable<float>("z"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(move, MsgSpatial3D, MsgNone) {
    const int ID = FLAMEGPU->getVariable<int>("id");
    const float REPULSE_FACTOR = FLAMEGPU->environment.getProperty<float>("repulse");
    const float RADIUS = FLAMEGPU->message_in.radius();
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");
    int count = 0;
    for (const auto &message : FLAMEGPU->message_in(x1, y1, z1)) {
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

// @todo - ad a way to visualise a single run of a single simulator somehow? maybe -v/--visualise <model_name> <pop>

// Run an individual simulation, using 
void run_circles_spatial3D(const simMethodParametrs params, simulationTiming &times){

    ModelDescription model(params.modelName);
    const float ENV_MAX = static_cast<float>(floor(cbrt(params.AGENT_COUNT)));
    {   // Location message
        MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
        message.newVariable<int>("id");
        message.setRadius(COMM_RADIUS);
        message.setMin(0, 0, 0);
        message.setMax(ENV_MAX, ENV_MAX, ENV_MAX);
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("Circle");
        agent.newVariable<int>("id");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("z");
        agent.newVariable<float>("drift");  // Store the distance moved here, for validation
        agent.newFunction("output_message", output_message).setMessageOutput("location");
        agent.newFunction("move", move).setMessageInput("location");
    }

    // Global environment variables.
    {
        EnvironmentDescription &env = model.Environment();
        env.newProperty("repulse", ENV_REPULSE);
    }

    // Organise the model. 

#if defined(CIRCLES_VALIDATION) && CIRCLES_VALIDATION
    {   // Attach init/step/exit functions and exit condition
        model.addStepFunction(Validation);
    }
#endif  // CIRCLES_VALIDATION

    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(output_message);
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(move);
    }

    // Create the simulation object
    CUDASimulation simulation(model);

    // Set config configuraiton properties 
    simulation.SimulationConfig().timing = false;
    simulation.SimulationConfig().verbose = false;
    simulation.SimulationConfig().random_seed = params.HOST_SEED;  // @todo device seed != host seed? 
    simulation.SimulationConfig().steps = params.STEPS;
    simulation.CUDAConfig().device_id = params.CUDA_DEVICE;

    // Generate the initial population
    std::default_random_engine rng(params.HOST_SEED);
    std::uniform_real_distribution<float> dist(0.0f, ENV_MAX);
    AgentVector population(model.Agent("Circle"), params.AGENT_COUNT);
    for (unsigned int i = 0; i < params.AGENT_COUNT; i++) {
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
    times.ms_rtc = simulation.getElapsedTimeRTCInitialisation();
    times.ms_simulation = simulation.getElapsedTimeSimulation();
    times.ms_init = simulation.getElapsedTimeInitFunctions();
    times.ms_exit = simulation.getElapsedTimeExitFunctions();
    
    std::vector<float> ms_steps = simulation.getElapsedTimeSteps();
    times.ms_per_step = std::make_shared<std::vector<float>>(std::vector<float>(ms_steps.begin(), ms_steps.end()));
    times.ms_stepMean = std::accumulate(ms_steps.begin(), ms_steps.end(), 0.f) / (float)simulation.getStepCounter();
}