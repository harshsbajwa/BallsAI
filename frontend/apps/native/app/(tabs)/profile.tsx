import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import React from "react";
import { View, Text, ScrollView, TouchableOpacity, Alert, ActivityIndicator } from "react-native";

import { authClient } from "@/lib/auth-client";
import { useAuth } from "@/lib/auth-provider";

export default function ProfileScreen() {
  const { data: session, isPending } = useAuth();
  const user = session?.user;
  const router = useRouter();

  const handleLogout = async () => {
    try {
      await authClient.signOut();
      Alert.alert("Logged Out", "You have been successfully logged out.");
    } catch (error) {
      Alert.alert("Error", "Failed to log out. Please try again.");
    }
  };

  if (isPending) {
      return (
          <View className="flex-1 bg-gray-50 justify-center items-center">
              <ActivityIndicator size="large" />
          </View>
      )
  }

  if (!user) {
    return (
      <View className="flex-1 bg-gray-50 justify-center items-center p-6">
        <Text className="text-2xl font-bold mb-4">Join the Action</Text>
        <Text className="text-gray-600 text-center mb-8">
          Log in or sign up to save your favorite players, teams, and get
          personalized content.
        </Text>
        <TouchableOpacity
          onPress={() => router.push("/(auth)/login")}
          className="bg-blue-600 w-full rounded-lg p-4 items-center justify-center mb-4"
        >
          <Text className="text-white font-bold text-base">Login</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => router.push("/(auth)/signup")}>
          <Text className="text-center text-blue-600">
            Don't have an account? Sign Up
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView className="flex-1 bg-gray-50">
      {/* Profile Header */}
      <View className="bg-blue-600 px-4 py-8">
        <View className="items-center">
          <View className="w-24 h-24 bg-white rounded-full items-center justify-center mb-4">
            <Ionicons name="person" size={50} color="#3B82F6" />
          </View>
          <Text className="text-white text-2xl font-bold">
            {user.email}
          </Text>
          <Text className="text-blue-100 text-base mt-1">
            Welcome to BallsAI
          </Text>
        </View>
      </View>

      {/* Settings */}
      <View className="p-4">
        <TouchableOpacity
          onPress={handleLogout}
          className="bg-white rounded-lg shadow-sm flex-row items-center p-4"
        >
          <View className="w-10 h-10 bg-red-100 rounded-full items-center justify-center mr-4">
            <Ionicons name="log-out-outline" size={22} color="#EF4444" />
          </View>
          <View className="flex-1">
            <Text className="font-semibold text-red-600 text-base">
              Logout
            </Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#9CA3AF" />
        </TouchableOpacity>

        <View className="mt-6 bg-white rounded-lg shadow-sm p-4">
          <Text className="font-semibold text-gray-900 mb-2">
            BallsAI Analytics
          </Text>
          <Text className="text-gray-500 text-sm">Version 1.0.0</Text>
        </View>
      </View>
    </ScrollView>
  );
}