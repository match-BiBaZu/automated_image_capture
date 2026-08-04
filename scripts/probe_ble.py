"""Read-only BLE diagnostic: list advertisements and optionally GATT services."""

from __future__ import annotations

import argparse
import asyncio

from bleak import BleakClient, BleakScanner


async def main(address: str | None) -> None:
    devices = await BleakScanner.discover(timeout=8.0, return_adv=True)
    print(f"Visible BLE devices: {len(devices)}")
    for device, advertisement in devices.values():
        print(
            f"{device.name or '<unnamed>'} {device.address} "
            f"RSSI={advertisement.rssi} services={advertisement.service_uuids}"
        )
    if address:
        async with BleakClient(address, timeout=8.0) as client:
            print(f"GATT services for {address}:")
            for service in client.services:
                print(f"  {service.uuid} {service.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", metavar="ADDRESS", help="Read GATT services from one device")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.connect))
