using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using SocketIO;
using UnityStandardAssets.Vehicles.Car;
using System;
using System.Security.AccessControl;

public class CommandServer : MonoBehaviour
{
    public CarRemoteControl CarRemoteControl;
    public Camera FrontFacingCamera;
    private SocketIOComponent _socket;
    private CarController _carController;

    // Use this for initialization
    void Start()
    {
        Debug.Log("Command server has been started");
        _socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
        _socket.On("open", OnOpen);
        _socket.On("move", OnMove);
        _carController = CarRemoteControl.GetComponent<CarController>();
        CarRemoteControl.SteeringAngle = 0f;
        CarRemoteControl.Acceleration = 0f;
    }

    void OnOpen(SocketIOEvent obj)
    {
        Debug.Log("Connection Open");
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            // Collect Data from the Car
            Dictionary<string, string> data = new Dictionary<string, string>();
            data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
            data["throttle"] = _carController.AccelInput.ToString("N4");
            data["speed"] = _carController.CurrentSpeed.ToString("N4");
            _socket.Emit("telemetry", new JSONObject(data));
        });

    }

    // Update is called once per frame
    void FixedUpdate()
    {
        Dictionary<string, string> data = new Dictionary<string, string>();
        data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
        data["throttle"] = _carController.AccelInput.ToString("N4");
        data["speed"] = _carController.CurrentSpeed.ToString("N4");
        data["image"] = Convert.ToBase64String(CaptureFrame(FrontFacingCamera));
        _socket.Emit("telemetry", new JSONObject(data));
    }

    void OnMove(SocketIOEvent obj)
    {
        JSONObject jsonObject = obj.data;
        CarRemoteControl.SteeringAngle = float.Parse(jsonObject.GetField("steering_angle").str);
        CarRemoteControl.Acceleration = float.Parse(jsonObject.GetField("throttle").str);
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            // Collect Data from the Car
            Dictionary<string, string> data = new Dictionary<string, string>();
            data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
            data["throttle"] = _carController.AccelInput.ToString("N4");
            data["speed"] = _carController.CurrentSpeed.ToString("N4");
            data["image"] = Convert.ToBase64String(CaptureFrame(FrontFacingCamera));
            _socket.Emit("telemetry", new JSONObject(data));
        });
    }

    byte[] CaptureFrame(Camera camera)
    {
        RenderTexture targetTexture = camera.targetTexture;
        RenderTexture.active = targetTexture;
        Texture2D texture2D = new Texture2D(targetTexture.width, targetTexture.height, TextureFormat.RGB24, false);
        texture2D.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), 0, 0);
        texture2D.Apply();
        byte[] image = texture2D.EncodeToJPG();
        UnityEngine.Object.DestroyImmediate(texture2D); // Required to prevent leaking the texture
        return image;
    }
}