#pragma once

#include <functional>

enum class EActivationType {
    PRESS_ANY,
    PRESS_ONCE,
    RELEASE_ONCE,
};

using EKey = int;
using EKeyCallback = std::function<void(float)>;

/**
 * Class managing keyboard events, detecting them and calling certain callbacks when they occur.
 * This can safely be instantiated multiple times, handling different events across different instances.
 */
class KeyManager {
    struct GLFWwindow *window = nullptr;

    using KeyCallbackInfo = std::pair<EActivationType, EKeyCallback>;
    std::unordered_map<EKey, KeyCallbackInfo> callbackMap;

    enum class KeyState {
        PRESSED,
        RELEASED
    };

    std::unordered_map<EKey, KeyState> keyStateMap;

public:
    explicit KeyManager(GLFWwindow *w) : window(w) {}

    /**
     * Binds a given callback to a keyboard event. Only one callback can be bound at a time,
     * so this will overwrite an earlier bound callback if there was any.
     *
     * @param k Key which on press should fire the callback.
     * @param type The way the key should be managed.
     * @param f The callback.
     */
    void bindCallback(EKey k, EActivationType type, const EKeyCallback& f);

    void tick(float deltaTime);

private:
    /**
     * Checks if a given keyboard event has occured.
     *
     * @param key Key to check.
     * @param type Type of event the caller is interested in.
     * @return Did the event occur?
     */
    bool checkKey(EKey key, EActivationType type);
};
