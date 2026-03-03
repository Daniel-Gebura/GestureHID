import asyncio

from ble_communication import BleHidProxyClient, BleHidProxyConfig, DeviceTarget


async def main():
    client = BleHidProxyClient(
        target=DeviceTarget(name="HID-Proxy-Control"),
        token="CHANGE_ME",
        config=BleHidProxyConfig(
            scan_timeout_s=10.0,
            write_retries=5,
            write_retry_delay_s=0.5,
            default_ack_timeout_s=1.0,
        ),
        on_log=print,  # optional
    )

    await client.connect()

    # Type a message on the TARGET PC (the one your nRF is plugged into)
    await client.keyboard_write("Hello from API client!\n")

    # Ctrl+A then backspace
    await client.keyboard_press(["CTRL"])
    await client.keyboard_press(["A"])
    await client.keyboard_release(["A"])
    await client.keyboard_release(["CTRL"])
    await client.keyboard_press(["BACKSPACE"])
    await client.keyboard_release(["BACKSPACE"])

    # Mouse move + click
    await client.mouse_move(dx=20, dy=0, wheel=0, require_ack=False)
    await client.mouse_click("LMB")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())