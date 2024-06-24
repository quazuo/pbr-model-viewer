struct WindowRes {
    uint width;
    uint height;
};

struct Matrices {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverse_vp;
    mat4 static_view;
};

struct MiscData {
    uint use_ibl;
    vec3 camera_pos;
    vec3 light_direction;
};
