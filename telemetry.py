import socket
import vpython as vp

print("==================================================")
print("       3D TELEMETRY GROUND STATION STARTING       ")
print("==================================================")

vp.scene.title = "<h2>Live Blimp Telemetry (Digital Twin)</h2>"
vp.scene.width = 1000
vp.scene.height = 550
vp.scene.background = vp.color.gray(0.1) 

vp.scene.append_to_caption("\n<hr>\n<h3>FLIGHT DASHBOARD</h3>\n")
dashboard = vp.wtext(text="<span style='color:orange;'>Status: AWAITING SIGNAL...</span><br><br><b>Pitch:</b> -- <br><b>Roll:</b>  -- <br><b>Yaw:</b>   --")
vp.scene.append_to_caption("\n<hr>\n<i><b>CONTROLS:</b> Hold the <b>CTRL</b> key + Left-Click and drag to rotate the camera. Scroll to zoom.</i>\n")

envelope = vp.ellipsoid(pos=vp.vector(0, 8, 0), length=24, height=12, width=24, 
                        color=vp.color.cyan, opacity=0.4)

pi_board = vp.box(pos=vp.vector(0, 0, 0), length=3, height=0.5, width=1, color=vp.color.blue)
carriage = vp.box(pos=vp.vector(0, -0.3, 0), length=3.2, height=0.1, width=1.2, color=vp.color.green)

beam = vp.box(pos=vp.vector(1, -0.8, 0), length=0.25, height=0.25, width=3, color=vp.color.gray(0.5))
motor_L = vp.cylinder(pos=vp.vector(1, -0.8, -1.5), axis=vp.vector(1.5, 0, 0), radius=0.25, color=vp.color.gray(0.7))
motor_R = vp.cylinder(pos=vp.vector(1, -0.8, 1.5), axis=vp.vector(1.5, 0, 0), radius=0.25, color=vp.color.gray(0.7))
motor_H = vp.cylinder(pos=vp.vector(-0.3, -0.4, 0), axis=vp.vector(0, -1.5, 0), radius=0.25, color=vp.color.gray(0.7))

blimp = vp.compound([envelope, pi_board, carriage, beam, motor_L, motor_R, motor_H])

UDP_IP = "0.0.0.0"  
UDP_PORT = 5005     

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False) 

print(f"Listening for IMU Telemetry on Port {UDP_PORT}...")

while True:
    vp.rate(60) 
    
    try:
        data, addr = sock.recvfrom(1024)
        telemetry = data.decode('utf-8').strip()
        
        parts = telemetry.split(',')
        if len(parts) == 3:
            pitch_deg = float(parts[0])
            roll_deg = float(parts[1])
            yaw_deg = float(parts[2])
            
            # FIXED: Closed the incomplete f-string literal and dashboard layout safely
            dashboard.text = f"<span style='color:lime;'>Status: LIVE CONNECTION</span><br><br><b>Pitch:</b> {pitch_deg:>6.1f}° <br><b>Roll:</b>  {roll_deg:>6.1f}° <br><b>Yaw:</b>   {yaw_deg:>6.1f}°"
            
            pitch_rad = vp.radians(pitch_deg)
            roll_rad = vp.radians(roll_deg)
            yaw_rad = vp.radians(yaw_deg)
            
            blimp.axis = vp.vector(vp.cos(pitch_rad)*vp.cos(yaw_rad), vp.sin(pitch_rad), vp.cos(pitch_rad)*vp.sin(yaw_rad))
            blimp.up = vp.vector(-vp.sin(roll_rad), vp.cos(roll_rad), 0)

    except BlockingIOError:
        pass
    except Exception as e:
        pass
